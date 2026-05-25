import argparse
import html
import os
import re
import subprocess
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import pysrt
from dotenv import load_dotenv
from google import genai

load_dotenv()


DEFAULT_CHUNK_SECONDS = 20
DEFAULT_CHUNK_OVERLAP_SECONDS = 2
DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"
DEFAULT_TRANSLATION_BATCH_SIZE = 30
DEFAULT_TRANSLATION_INTERVAL_SECONDS = 1.0
DEFAULT_TRANSLATION_RETRIES = 2
DEFAULT_ASR_MODEL = "nvidia/parakeet-tdt_ctc-0.6b-ja"
MERGE_GAP_SECONDS = 0.3
AUDIO_CODEC = "pcm_s16le"
AUDIO_SAMPLE_RATE = "16000"
AUDIO_CHANNELS = "1"


class SubtitleSegment(TypedDict):
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class AudioChunk:
    path: Path
    start: float
    end: float
    keep_start: float
    keep_end: float


def run_command(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, capture_output=True, text=True, check=True)
    except FileNotFoundError as e:
        raise RuntimeError(f"Required command not found: {command[0]}") from e
    except subprocess.CalledProcessError as e:
        detail = (e.stderr or e.stdout or "").strip()
        message = f"{command[0]} failed with exit code {e.returncode}"
        if detail:
            message = f"{message}: {detail}"
        raise RuntimeError(message) from e


def ffmpeg_audio_args() -> list[str]:
    return ["-acodec", AUDIO_CODEC, "-ar", AUDIO_SAMPLE_RATE, "-ac", AUDIO_CHANNELS]


def get_audio_duration(audio_path: Path) -> float:
    result = run_command([
        "ffprobe",
        "-v",
        "quiet",
        "-show_entries",
        "format=duration",
        "-of",
        "csv=p=0",
        str(audio_path),
    ])
    raw_duration = result.stdout.strip()
    try:
        duration = float(raw_duration)
    except ValueError as e:
        raise RuntimeError(f"Could not read duration for {audio_path}: {raw_duration!r}") from e
    if duration <= 0:
        raise RuntimeError(f"Audio duration must be greater than 0: {audio_path}")
    return duration


def extract_audio(video_path: Path, audio_path: Path) -> None:
    run_command([
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vn",
        *ffmpeg_audio_args(),
        str(audio_path),
        "-y",
    ])


def split_audio(
    audio_path: Path, chunk_dir: Path, chunk_seconds: float, chunk_overlap_seconds: float
) -> list[AudioChunk]:
    duration = get_audio_duration(audio_path)
    chunk_step = chunk_seconds - chunk_overlap_seconds
    if chunk_step <= 0:
        raise ValueError("chunk_overlap_seconds must be smaller than chunk_seconds")

    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[AudioChunk] = []
    start = 0.0
    idx = 0
    while start < duration:
        chunk_path = chunk_dir / f"chunk_{idx:04d}.wav"
        chunk_duration = min(chunk_seconds, duration - start)
        chunk_end = start + chunk_duration
        run_command([
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(audio_path),
            "-ss",
            str(start),
            "-t",
            str(chunk_duration),
            *ffmpeg_audio_args(),
            str(chunk_path),
            "-y",
        ])
        keep_end = min(start + chunk_step, duration)
        if chunk_end >= duration:
            keep_end = duration

        chunks.append(
            AudioChunk(
                path=chunk_path,
                start=start,
                end=chunk_end,
                keep_start=start,
                keep_end=keep_end,
            )
        )
        start += chunk_step
        idx += 1
    return chunks


def format_timestamp(seconds: float) -> str:
    total_ms = int(max(0.0, seconds) * 1000 + 0.5)
    ms = total_ms % 1000
    total_seconds = total_ms // 1000
    s = total_seconds % 60
    total_minutes = total_seconds // 60
    m = total_minutes % 60
    h = total_minutes // 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def keep_segment_in_window(segment: SubtitleSegment, keep_start: float, keep_end: float) -> SubtitleSegment | None:
    segment_start = segment["start"]
    if keep_start <= segment_start < keep_end:
        return segment
    return None


def merge_adjacent_segments(
    segments: list[SubtitleSegment], merge_gap_seconds: float = MERGE_GAP_SECONDS
) -> list[SubtitleSegment]:
    if not segments:
        return []

    sorted_segments = sorted(segments, key=lambda seg: (seg["start"], seg["end"]))
    merged = [sorted_segments[0].copy()]

    for segment in sorted_segments[1:]:
        prev = merged[-1]
        prev_text = prev["text"].strip()
        current_text = segment["text"].strip()
        gap = segment["start"] - prev["end"]

        if current_text == prev_text:
            prev["end"] = max(prev["end"], segment["end"])
            continue

        if gap <= merge_gap_seconds and prev_text and current_text and current_text.startswith(prev_text):
            prev["text"] = current_text
            prev["end"] = segment["end"]
            continue

        merged.append(segment.copy())

    return merged


def transcribe(
    audio_path: Path,
    tmp_dir: Path,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
    chunk_overlap_seconds: float = DEFAULT_CHUNK_OVERLAP_SECONDS,
) -> list[SubtitleSegment]:
    import nemo.collections.asr as nemo_asr
    import torch
    from nemo.collections.asr.parts.mixins import TranscribeConfig

    # 1. 检查并设置设备

    # Apple Silicon 支持 mps 加速，Intel 芯片则使用 cpu
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✨ 检测到 Apple Silicon，使用 MPS 硬件加速")
    else:
        device = torch.device("cpu")
        print("💻 使用 CPU 运行")

    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=DEFAULT_ASR_MODEL)

    # 将模型移至指定设备并设为评估模式
    asr_model = asr_model.to(device)  # type: ignore
    asr_model.eval()

    chunk_dir = tmp_dir / "chunks"
    chunks = split_audio(audio_path, chunk_dir, chunk_seconds, chunk_overlap_seconds)
    print(f"Split audio into {len(chunks)} chunks of {chunk_seconds}s each with {chunk_overlap_seconds}s overlap")

    all_segments: list[SubtitleSegment] = []
    for i, chunk in enumerate(chunks):
        print(f"  Transcribing chunk {i + 1}/{len(chunks)}...")
        with torch.no_grad():
            output = asr_model.transcribe(
                [str(chunk.path)],
                timestamps=True,
                override_config=TranscribeConfig(
                    use_lhotse=False,
                    batch_size=1,
                    num_workers=0,
                    timestamps=True,
                    verbose=False,
                ),
            )

        for hypothesis in output:
            ts = hypothesis.timestamp
            if ts and ts.get("char"):
                chars = ts["char"]
                for seg in group_chars_into_segments(chars, chunk.start):
                    kept_seg = keep_segment_in_window(seg, chunk.keep_start, chunk.keep_end)
                    if kept_seg is not None:
                        all_segments.append(kept_seg)
            elif ts and ts.get("segment"):
                for seg in ts["segment"]:
                    segment: SubtitleSegment = {
                        "start": float(seg["start"]) + chunk.start,
                        "end": float(seg["end"]) + chunk.start,
                        "text": str(seg["segment"]),
                    }
                    kept_seg = keep_segment_in_window(segment, chunk.keep_start, chunk.keep_end)
                    if kept_seg is not None:
                        all_segments.append(kept_seg)
            elif hypothesis.text:
                all_segments.append({
                    "start": chunk.keep_start,
                    "end": chunk.keep_end,
                    "text": hypothesis.text,
                })

    return merge_adjacent_segments(all_segments)


def group_chars_into_segments(
    chars: list[dict], time_offset: float = 0.0, max_duration: float = 5.0, max_chars: int = 40
) -> list[SubtitleSegment]:
    segments: list[SubtitleSegment] = []
    current_text = ""
    current_start: float | None = None
    current_end = 0.0

    for item in chars:
        text = "".join(item["char"])
        item_start = float(item["start"])
        item_end = float(item["end"])

        if current_start is None:
            current_start = item_start
            current_text = text
            current_end = item_end
            continue

        duration = item_end - current_start
        if duration > max_duration or len(current_text) + len(text) > max_chars:
            segments.append({
                "start": current_start + time_offset,
                "end": current_end + time_offset,
                "text": current_text,
            })
            current_start = item_start
            current_text = text
        else:
            current_text += text
        current_end = item_end

    if current_text and current_start is not None:
        segments.append({
            "start": current_start + time_offset,
            "end": current_end + time_offset,
            "text": current_text,
        })

    return segments


def write_srt(segments: list[SubtitleSegment], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            start = format_timestamp(seg["start"])
            end = format_timestamp(seg["end"])
            f.write(f"{i}\n{start} --> {end}\n{seg['text']}\n\n")


def get_gemini_model_name(model_name: str | None = None) -> str:
    return model_name or os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL


def default_japanese_srt_path(media_path: Path) -> Path:
    if media_path.suffix:
        return media_path.with_name(f"{media_path.stem}.jp.srt")
    return media_path.with_name(f"{media_path.name}.jp.srt")


def default_chinese_srt_path(srt_path: Path) -> Path:
    if srt_path.suffix == ".srt" and srt_path.stem.endswith(".jp"):
        return srt_path.with_name(f"{srt_path.stem[:-3]}.srt")
    if srt_path.suffix:
        return srt_path.with_name(f"{srt_path.stem}.translated{srt_path.suffix}")
    return srt_path.with_name(f"{srt_path.name}.translated.srt")


def generate_translation(client: genai.Client, model_name: str, prompt: str) -> str:
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )
    text = response.text
    if text is None:
        raise RuntimeError("Gemini returned an empty response")
    return text.strip()


def build_translation_prompt(texts: list[str]) -> str:
    tagged_lines = "\n".join(
        f'<line id="{idx}">{html.escape(text, quote=False)}</line>' for idx, text in enumerate(texts, 1)
    )
    return (
        "Translate these Japanese subtitle lines into natural, concise Simplified Chinese.\n"
        "Return exactly one translated line for every input line.\n"
        "Keep the same ids and output only XML-like lines in this exact format:\n"
        '<line id="1">translated text</line>\n\n'
        f"{tagged_lines}"
    )


def parse_translation_lines(response_text: str, expected_count: int) -> list[str] | None:
    translations: dict[int, str] = {}
    for match in re.finditer(r'<line\s+id="(\d+)">\s*(.*?)\s*</line>', response_text, re.DOTALL):
        idx = int(match.group(1))
        if idx in translations:
            return None
        translations[idx] = " ".join(html.unescape(match.group(2)).split())

    if sorted(translations) != list(range(1, expected_count + 1)):
        return None

    return [translations[idx] for idx in range(1, expected_count + 1)]


def get_retry_delay_seconds(error: Exception) -> float | None:
    message = str(error)
    retry_delay_match = re.search(r"retryDelay['\"]?:\s*['\"]?(\d+(?:\.\d+)?)s", message)
    if retry_delay_match:
        return float(retry_delay_match.group(1))

    retry_in_match = re.search(r"retry in (\d+(?:\.\d+)?)s", message, re.IGNORECASE)
    if retry_in_match:
        return float(retry_in_match.group(1))

    return None


def wait_between_requests(previous_request_time: float | None, interval_seconds: float) -> None:
    if previous_request_time is None or interval_seconds <= 0:
        return

    elapsed = time.monotonic() - previous_request_time
    wait_seconds = interval_seconds - elapsed
    if wait_seconds > 0:
        print(f"Waiting {wait_seconds:.1f}s to respect Gemini rate limits...")
        time.sleep(wait_seconds)


def translate_batch(
    client: genai.Client,
    model_name: str,
    texts: list[str],
    previous_request_time: float | None,
    request_interval_seconds: float,
    retries: int,
) -> tuple[list[str], float]:
    prompt = build_translation_prompt(texts)
    last_error: Exception | None = None

    for attempt in range(retries + 1):
        try:
            wait_between_requests(previous_request_time, request_interval_seconds)
            request_time = time.monotonic()
            response_text = generate_translation(client, model_name, prompt)
            translations = parse_translation_lines(response_text, len(texts))
            if translations is None:
                raise ValueError(f"Expected {len(texts)} tagged translations, but Gemini returned a different shape.")
            return translations, request_time
        except Exception as e:
            last_error = e
            if attempt >= retries:
                break

            retry_delay = get_retry_delay_seconds(e)
            if retry_delay is not None:
                wait_seconds = retry_delay + 1
                print(f"Gemini rate limit hit. Waiting {wait_seconds:.1f}s before retrying...")
                time.sleep(wait_seconds)
            else:
                wait_seconds = min(2**attempt, 8)
                print(f"Translation batch shape/error: {e}. Retrying in {wait_seconds:.1f}s...")
                time.sleep(wait_seconds)

    raise RuntimeError(f"Translation batch failed after {retries + 1} attempt(s): {last_error}")


def translate_subtitles(
    input_path: Path,
    output_path: Path,
    model_name: str | None = None,
    batch_size: int = DEFAULT_TRANSLATION_BATCH_SIZE,
    request_interval_seconds: float = DEFAULT_TRANSLATION_INTERVAL_SECONDS,
) -> None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Warning: GEMINI_API_KEY not found in .env file. Skipping translation.")
        return
    if batch_size <= 0:
        raise SystemExit("--batch-size must be greater than 0")
    if request_interval_seconds < 0:
        raise SystemExit("--request-interval-seconds must be 0 or greater")

    model_name = get_gemini_model_name(model_name)
    client = genai.Client(api_key=api_key, http_options={"api_version": "v1"})

    subs = pysrt.open(input_path, encoding="utf-8")
    print(f"Translating {len(subs)} segments to Chinese with {model_name}...")

    failed_batches = 0
    previous_request_time: float | None = None
    for i in range(0, len(subs), batch_size):
        batch = subs[i : i + batch_size]
        texts = [s.text for s in batch]
        batch_number = (i // batch_size) + 1
        batch_count = (len(subs) + batch_size - 1) // batch_size
        print(f"  Translating batch {batch_number}/{batch_count} ({len(batch)} lines)...")

        try:
            translated_lines, previous_request_time = translate_batch(
                client,
                model_name,
                texts,
                previous_request_time,
                request_interval_seconds,
                DEFAULT_TRANSLATION_RETRIES,
            )
            for sub, translated_text in zip(batch, translated_lines):
                sub.text = translated_text.strip()
        except Exception as e:
            failed_batches += 1
            print(f"Error during translation batch starting at {i}: {e}")
            continue

    if failed_batches:
        raise SystemExit(f"Translation failed for {failed_batches} batch(es); output was not saved.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    subs.save(output_path, encoding="utf-8")
    print(f"Translated subtitles saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Transcribe Japanese audio from MP4 to SRT subtitles")
    parser.add_argument("input", type=Path, help="Input MP4 video file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output Japanese SRT file (default: same name as input with .jp.srt extension)",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=DEFAULT_CHUNK_SECONDS,
        help=f"Chunk duration in seconds (default: {DEFAULT_CHUNK_SECONDS})",
    )
    parser.add_argument(
        "--chunk-overlap-seconds",
        type=float,
        default=DEFAULT_CHUNK_OVERLAP_SECONDS,
        help=f"Overlap between chunks in seconds (default: {DEFAULT_CHUNK_OVERLAP_SECONDS})",
    )
    parser.add_argument(
        "--translation-model",
        default=None,
        help=f"Gemini model for Chinese translation (default: GEMINI_MODEL or {DEFAULT_GEMINI_MODEL})",
    )
    parser.add_argument(
        "--translation-batch-size",
        type=int,
        default=DEFAULT_TRANSLATION_BATCH_SIZE,
        help=f"Subtitle lines per translation request (default: {DEFAULT_TRANSLATION_BATCH_SIZE})",
    )
    parser.add_argument(
        "--translation-request-interval-seconds",
        type=float,
        default=DEFAULT_TRANSLATION_INTERVAL_SECONDS,
        help=f"Minimum delay between Gemini requests (default: {DEFAULT_TRANSLATION_INTERVAL_SECONDS})",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")
    if args.chunk_seconds <= 0:
        raise SystemExit("--chunk-seconds must be greater than 0")
    if args.chunk_overlap_seconds < 0:
        raise SystemExit("--chunk-overlap-seconds must be 0 or greater")
    if args.chunk_overlap_seconds >= args.chunk_seconds:
        raise SystemExit("--chunk-overlap-seconds must be smaller than --chunk-seconds")

    output_path = args.output or default_japanese_srt_path(args.input)
    audio_path = args.input.with_suffix(".wav")

    if audio_path.exists():
        print(f"Using existing audio file: {audio_path}")
    else:
        print(f"Extracting audio from {args.input}...")
        extract_audio(args.input, audio_path)

    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"Using temporary directory: {tmp_dir}")
        print(f"Transcribing with {DEFAULT_ASR_MODEL}...")
        segments = transcribe(
            audio_path,
            Path(tmp_dir),
            chunk_seconds=args.chunk_seconds,
            chunk_overlap_seconds=args.chunk_overlap_seconds,
        )

    print(f"Writing {len(segments)} subtitle segments to {output_path}")
    write_srt(segments, output_path)

    # Translate to Chinese
    output_cn_path = default_chinese_srt_path(output_path)
    translate_subtitles(
        output_path,
        output_cn_path,
        args.translation_model,
        args.translation_batch_size,
        args.translation_request_interval_seconds,
    )

    print("Done.")


def translate_cli():
    parser = argparse.ArgumentParser(description="Translate an existing SRT file to Chinese using Gemini")
    parser.add_argument("input", type=Path, help="Input SRT file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output Chinese SRT file (default: .jp.srt -> .srt, otherwise append .translated before .srt)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"Gemini model to use (default: GEMINI_MODEL or {DEFAULT_GEMINI_MODEL})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_TRANSLATION_BATCH_SIZE,
        help=f"Subtitle lines per translation request (default: {DEFAULT_TRANSLATION_BATCH_SIZE})",
    )
    parser.add_argument(
        "--request-interval-seconds",
        type=float,
        default=DEFAULT_TRANSLATION_INTERVAL_SECONDS,
        help=f"Minimum delay between Gemini requests (default: {DEFAULT_TRANSLATION_INTERVAL_SECONDS})",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    output_path = args.output or default_chinese_srt_path(args.input)
    translate_subtitles(args.input, output_path, args.model, args.batch_size, args.request_interval_seconds)
    print("Done.")


if __name__ == "__main__":
    main()
