import argparse
import html
import math
import os
import platform
import re
import subprocess
import tempfile
import time
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import pysrt
from dotenv import load_dotenv
from google.genai import Client

load_dotenv()


DEFAULT_CHUNK_SECONDS = 20
DEFAULT_CHUNK_OVERLAP_SECONDS = 2
DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"
DEFAULT_TRANSLATION_BATCH_SIZE = 50
DEFAULT_TRANSLATION_INTERVAL_SECONDS = 1.0
DEFAULT_TRANSLATION_RETRIES = 2
DEFAULT_NEMO_ASR_MODEL = "nvidia/parakeet-tdt_ctc-0.6b-ja"
DEFAULT_MLX_ASR_MODEL = "mlx-community/parakeet-tdt_ctc-0.6b-ja"
DEFAULT_ASR_MODEL = (
    DEFAULT_MLX_ASR_MODEL if platform.system() == "Darwin" and platform.machine() == "arm64" else DEFAULT_NEMO_ASR_MODEL
)
MERGE_GAP_SECONDS = 0.3
DEFAULT_MAX_SEGMENT_DURATION_SECONDS = 20.0
DEFAULT_MAX_SEGMENT_CHARS = 45
DEFAULT_SEGMENT_SILENCE_GAP_SECONDS = 0.8
MIN_SPLIT_SEGMENT_CHARS = 4
AUDIO_CODEC = "pcm_s16le"
AUDIO_SAMPLE_RATE = "16000"
AUDIO_CHANNELS = "1"
HARD_BREAK_CHARS = "。．.!！?？\n"
SOFT_BREAK_CHARS = "、，,；;：: "


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
        keep_start = start
        keep_end = duration if chunk_end >= duration else min(start + chunk_step, duration)

        chunks.append(
            AudioChunk(
                path=chunk_path,
                start=start,
                end=chunk_end,
                keep_start=keep_start,
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
    segment_midpoint = (segment["start"] + segment["end"]) / 2
    if keep_start <= segment_midpoint < keep_end:
        return segment
    return None


def is_punctuation_only(text: str) -> bool:
    compact_text = "".join(text.split())
    return bool(compact_text) and all(unicodedata.category(char).startswith("P") for char in compact_text)


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

        if prev_text and is_punctuation_only(current_text):
            prev["text"] = f"{prev_text}{current_text}"
            if gap <= merge_gap_seconds:
                prev["end"] = max(prev["end"], segment["end"])
            continue

        if gap <= merge_gap_seconds and current_text == prev_text:
            prev["end"] = max(prev["end"], segment["end"])
            continue

        if gap <= merge_gap_seconds and prev_text and current_text and current_text.startswith(prev_text):
            prev["text"] = current_text
            prev["end"] = segment["end"]
            continue

        merged.append(segment.copy())

    return merged


def choose_split_index(text: str, start: int, ideal_end: int, max_end: int) -> int:
    ideal_end = min(len(text), max(start + 1, ideal_end))
    max_end = min(len(text), max(start + 1, max_end))
    if max_end >= len(text):
        return len(text)

    for break_chars in (HARD_BREAK_CHARS, SOFT_BREAK_CHARS):
        for idx in range(ideal_end, start, -1):
            if text[idx - 1] in break_chars:
                return idx
        for idx in range(ideal_end + 1, min(len(text), max_end + 1)):
            if text[idx - 1] in break_chars:
                return idx

    return ideal_end


def split_text_for_subtitles(text: str, max_chars: int, target_parts: int) -> list[str]:
    clean_text = text.strip()
    if not clean_text:
        return []
    target_parts = min(len(clean_text), max(1, target_parts))
    if len(clean_text) <= max_chars and target_parts <= 1:
        return [clean_text]

    parts: list[str] = []
    start = 0
    remaining_parts = target_parts
    while start < len(clean_text):
        remaining_chars = len(clean_text) - start
        if remaining_parts <= 1 or remaining_chars <= 1:
            parts.append(clean_text[start:].strip())
            break

        max_chunk_chars = min(max_chars, remaining_chars - (remaining_parts - 1))
        soft_limit = max(1, min(max_chunk_chars, math.ceil(remaining_chars / remaining_parts)))
        split_idx = choose_split_index(
            clean_text,
            start,
            start + soft_limit,
            start + max_chunk_chars,
        )
        if split_idx <= start:
            split_idx = min(len(clean_text), start + max_chunk_chars)

        part = clean_text[start:split_idx].strip()
        if not part:
            split_idx = min(len(clean_text), start + max_chunk_chars)
            part = clean_text[start:split_idx].strip()

        parts.append(part)
        start = split_idx
        remaining_parts = max(1, remaining_parts - 1)

    return [part for part in parts if part]


def split_long_segment(segment: SubtitleSegment, max_duration: float, max_chars: int) -> list[SubtitleSegment]:
    text = segment["text"].strip()
    duration = segment["end"] - segment["start"]
    if not text:
        return []
    if duration <= max_duration and len(text) <= max_chars:
        return [segment]

    target_parts = max(
        math.ceil(duration / max_duration) if max_duration > 0 else 1,
        math.ceil(len(text) / max_chars) if max_chars > 0 else 1,
    )
    target_parts = min(target_parts, max(1, len(text) // MIN_SPLIT_SEGMENT_CHARS))
    parts = split_text_for_subtitles(text, max_chars=max_chars, target_parts=target_parts)
    if len(parts) <= 1:
        return [segment]

    total_chars = sum(len(part) for part in parts)
    if total_chars <= 0 or duration <= 0:
        part_duration = duration / len(parts) if parts else 0.0
        return [
            {
                "start": segment["start"] + (idx * part_duration),
                "end": segment["start"] + ((idx + 1) * part_duration),
                "text": part,
            }
            for idx, part in enumerate(parts)
        ]

    split_segments: list[SubtitleSegment] = []
    current_start = segment["start"]
    consumed_chars = 0
    for idx, part in enumerate(parts):
        consumed_chars += len(part)
        if idx == len(parts) - 1:
            current_end = segment["end"]
        else:
            current_end = segment["start"] + (duration * consumed_chars / total_chars)
        split_segments.append({
            "start": current_start,
            "end": current_end,
            "text": part,
        })
        current_start = current_end

    return split_segments


def normalize_segments(
    segments: list[SubtitleSegment],
    max_segment_duration: float,
    max_segment_chars: int,
) -> list[SubtitleSegment]:
    merged_segments = merge_adjacent_segments(segments)
    normalized_segments: list[SubtitleSegment] = []
    for segment in merged_segments:
        for split_segment in split_long_segment(segment, max_segment_duration, max_segment_chars):
            if split_segment["end"] - split_segment["start"] > max_segment_duration:
                split_segment = split_segment.copy()
                split_segment["end"] = split_segment["start"] + max_segment_duration
            normalized_segments.append(split_segment)
    return normalized_segments


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def is_mlx_asr_model(model_name: str) -> bool:
    return model_name.startswith("mlx-community/")


def get_asr_model_name(model_name: str | None = None) -> str:
    return model_name or os.getenv("PARAKEET_ASR_MODEL") or DEFAULT_ASR_MODEL


def get_hypothesis_timestamps(hypothesis: object) -> dict:
    # NeMo exposes decoded timestamps as `timestep`; some versions and model
    # families use `timestamp` instead.
    timestamp_dicts: list[dict] = []
    for attribute_name in ("timestep", "timestamp"):
        value = getattr(hypothesis, attribute_name, None)
        if isinstance(value, dict):
            timestamp_dicts.append(value)
            if any(value.get(level) for level in ("char", "segment", "word")):
                return value
    return timestamp_dicts[0] if timestamp_dicts else {}


def filter_chars_to_window(
    chars: list[dict], time_offset: float, keep_start: float, keep_end: float
) -> list[dict]:
    filtered: list[dict] = []
    for item in chars:
        item_start = float(item["start"]) + time_offset
        item_end = float(item["end"]) + time_offset
        midpoint = (item_start + item_end) / 2
        if keep_start <= midpoint < keep_end:
            filtered.append(item)
    return filtered


def transcribe_with_mlx(
    audio_path: Path,
    model_name: str,
    chunk_seconds: float,
    chunk_overlap_seconds: float,
    max_segment_duration: float,
    max_segment_chars: int,
    segment_silence_gap_seconds: float,
) -> list[SubtitleSegment]:
    if not is_apple_silicon():
        raise RuntimeError("MLX ASR models currently require Apple Silicon (macOS arm64).")

    try:
        from parakeet_mlx import (  # pyright: ignore[reportMissingImports]
            DecodingConfig,
            SentenceConfig,
            from_pretrained,
        )
    except ImportError as e:
        raise RuntimeError(
            "parakeet-mlx is not installed. Run `uv sync` on Apple Silicon to install the MLX backend."
        ) from e

    print("Apple Silicon detected, using MLX ASR backend")
    asr_model = from_pretrained(model_name)
    result = asr_model.transcribe(
        audio_path,
        decoding_config=DecodingConfig(
            sentence=SentenceConfig(
                silence_gap=segment_silence_gap_seconds,
                max_duration=max_segment_duration,
            )
        ),
        chunk_duration=chunk_seconds if chunk_seconds > 0 else None,
        overlap_duration=chunk_overlap_seconds,
    )

    segments: list[SubtitleSegment] = []
    for sentence in result.sentences:
        text = sentence.text.strip()
        if not text:
            continue
        segments.append({
            "start": float(sentence.start),
            "end": float(sentence.end),
            "text": text,
        })

    return normalize_segments(segments, max_segment_duration, max_segment_chars)


def transcribe_with_nemo(
    audio_path: Path,
    tmp_dir: Path,
    model_name: str,
    chunk_seconds: float,
    chunk_overlap_seconds: float,
    max_segment_duration: float,
    max_segment_chars: int,
    segment_silence_gap_seconds: float,
) -> list[SubtitleSegment]:
    if is_mlx_asr_model(model_name):
        raise RuntimeError("MLX models must be loaded with the MLX backend, not NeMo.")

    import nemo.collections.asr as nemo_asr  # pyright: ignore[reportMissingImports]
    import torch  # pyright: ignore[reportMissingImports]

    # Apple Silicon supports MPS; other machines fall back to CPU.
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Apple Silicon detected, using MPS acceleration")
    else:
        device = torch.device("cpu")
        print("Using CPU for NeMo transcription")

    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)

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
                batch_size=1,
                num_workers=0,
                verbose=False,
            )

        for hypothesis in output:
            ts = get_hypothesis_timestamps(hypothesis)
            if ts and ts.get("char"):
                chars = filter_chars_to_window(ts["char"], chunk.start, chunk.keep_start, chunk.keep_end)
                for seg in group_chars_into_segments(
                    chars,
                    chunk.start,
                    max_duration=max_segment_duration,
                    max_chars=max_segment_chars,
                    silence_gap_seconds=segment_silence_gap_seconds,
                ):
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
                raise RuntimeError(
                    f"ASR model returned text without usable timestamps for chunk {i + 1} "
                    f"({chunk.start:.3f}s-{chunk.end:.3f}s); refusing to create inaccurate subtitles."
                )

    return normalize_segments(all_segments, max_segment_duration, max_segment_chars)


def transcribe(
    audio_path: Path,
    tmp_dir: Path,
    model_name: str,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
    chunk_overlap_seconds: float = DEFAULT_CHUNK_OVERLAP_SECONDS,
    max_segment_duration: float = DEFAULT_MAX_SEGMENT_DURATION_SECONDS,
    max_segment_chars: int = DEFAULT_MAX_SEGMENT_CHARS,
    segment_silence_gap_seconds: float = DEFAULT_SEGMENT_SILENCE_GAP_SECONDS,
) -> list[SubtitleSegment]:
    if is_mlx_asr_model(model_name):
        return transcribe_with_mlx(
            audio_path,
            model_name,
            chunk_seconds,
            chunk_overlap_seconds,
            max_segment_duration,
            max_segment_chars,
            segment_silence_gap_seconds,
        )

    return transcribe_with_nemo(
        audio_path,
        tmp_dir,
        model_name,
        chunk_seconds,
        chunk_overlap_seconds,
        max_segment_duration,
        max_segment_chars,
        segment_silence_gap_seconds,
    )


def group_chars_into_segments(
    chars: list[dict],
    time_offset: float = 0.0,
    max_duration: float = DEFAULT_MAX_SEGMENT_DURATION_SECONDS,
    max_chars: int = DEFAULT_MAX_SEGMENT_CHARS,
    silence_gap_seconds: float = DEFAULT_SEGMENT_SILENCE_GAP_SECONDS,
) -> list[SubtitleSegment]:
    segments: list[SubtitleSegment] = []
    current_text = ""
    current_start: float | None = None
    current_end = 0.0

    for item in chars:
        text = "".join(item["char"])
        if not text:
            continue
        item_start = float(item["start"])
        item_end = float(item["end"])

        if current_start is None:
            current_start = item_start
            current_text = text
            current_end = item_end
            continue

        duration = item_end - current_start
        silence_gap = item_start - current_end
        if silence_gap >= silence_gap_seconds or duration > max_duration or len(current_text) + len(text) > max_chars:
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


def get_response_field(value: object, field_name: str) -> object | None:
    if isinstance(value, dict):
        return value.get(field_name)
    return getattr(value, field_name, None)


def get_response_items(value: object, field_name: str) -> list[object]:
    items = get_response_field(value, field_name)
    if items is None:
        return []
    if isinstance(items, list | tuple):
        return list(items)
    return [items]


def summarize_response_value(value: object, max_length: int = 200) -> str:
    text = str(value).replace("\n", " ").strip()
    if len(text) <= max_length:
        return text
    return f"{text[: max_length - 3]}..."


def extract_response_text(response: object) -> str | None:
    response_text = get_response_field(response, "text")
    if isinstance(response_text, str) and response_text.strip():
        return response_text.strip()

    texts: list[str] = []
    for candidate in get_response_items(response, "candidates"):
        content = get_response_field(candidate, "content")
        for part in get_response_items(content, "parts"):
            part_text = get_response_field(part, "text")
            if isinstance(part_text, str) and part_text.strip():
                texts.append(part_text.strip())

    if not texts:
        return None
    return "\n".join(texts).strip()


def describe_empty_translation_response(response: object) -> str:
    details: list[str] = []

    finish_reasons = [
        summarize_response_value(reason)
        for candidate in get_response_items(response, "candidates")
        for reason in [get_response_field(candidate, "finish_reason") or get_response_field(candidate, "finishReason")]
        if reason is not None
    ]
    if finish_reasons:
        details.append(f"finish_reason={', '.join(finish_reasons)}")

    prompt_feedback = get_response_field(response, "prompt_feedback") or get_response_field(response, "promptFeedback")
    if prompt_feedback is not None:
        details.append(f"prompt_feedback={summarize_response_value(prompt_feedback)}")

    if not details:
        return "no text parts or candidates were returned"
    return "; ".join(details)


def generate_translation(client: Client, model_name: str, prompt: str) -> str:
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )
    text = extract_response_text(response)
    if text is None:
        raise RuntimeError(f"Gemini returned an empty response ({describe_empty_translation_response(response)})")
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


def should_split_translation_batch(error: Exception) -> bool:
    message = str(error).lower()
    split_markers = [
        "empty response",
        "different shape",
        "finish_reason",
        "prompt_feedback",
        "safety",
        "blocked",
    ]
    return any(marker in message for marker in split_markers)


def translate_batch(
    client: Client,
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


def translate_batch_with_fallback(
    client: Client,
    model_name: str,
    texts: list[str],
    previous_request_time: float | None,
    request_interval_seconds: float,
    retries: int,
) -> tuple[list[str], float]:
    try:
        return translate_batch(
            client,
            model_name,
            texts,
            previous_request_time,
            request_interval_seconds,
            retries,
        )
    except Exception as e:
        if len(texts) > 1 and should_split_translation_batch(e):
            split_at = len(texts) // 2
            print(f"    Batch of {len(texts)} lines still failed; retrying as {split_at} + {len(texts) - split_at}...")
            left_translations, previous_request_time = translate_batch_with_fallback(
                client,
                model_name,
                texts[:split_at],
                previous_request_time,
                request_interval_seconds,
                retries,
            )
            right_translations, previous_request_time = translate_batch_with_fallback(
                client,
                model_name,
                texts[split_at:],
                previous_request_time,
                request_interval_seconds,
                retries,
            )
            return left_translations + right_translations, previous_request_time

        if len(texts) == 1 and should_split_translation_batch(e):
            raise RuntimeError(f"Gemini could not translate subtitle line: {texts[0]!r}: {e}") from e

        raise


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
    client = Client(api_key=api_key, http_options={"api_version": "v1"})

    subs = pysrt.open(str(input_path), encoding="utf-8")
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
            translated_lines, previous_request_time = translate_batch_with_fallback(
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
        "--max-segment-duration-seconds",
        type=float,
        default=DEFAULT_MAX_SEGMENT_DURATION_SECONDS,
        help=(
            "Maximum subtitle segment duration after post-processing "
            f"(default: {DEFAULT_MAX_SEGMENT_DURATION_SECONDS})"
        ),
    )
    parser.add_argument(
        "--max-segment-chars",
        type=int,
        default=DEFAULT_MAX_SEGMENT_CHARS,
        help=f"Maximum subtitle characters per segment after post-processing (default: {DEFAULT_MAX_SEGMENT_CHARS})",
    )
    parser.add_argument(
        "--segment-silence-gap-seconds",
        type=float,
        default=DEFAULT_SEGMENT_SILENCE_GAP_SECONDS,
        help=(
            "Split subtitle segments when silence exceeds this threshold "
            f"and char timestamps are available (default: {DEFAULT_SEGMENT_SILENCE_GAP_SECONDS})"
        ),
    )
    parser.add_argument(
        "--asr-model",
        default=None,
        help=f"ASR model to use (default: PARAKEET_ASR_MODEL or {DEFAULT_ASR_MODEL})",
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
    if args.max_segment_duration_seconds <= 0:
        raise SystemExit("--max-segment-duration-seconds must be greater than 0")
    if args.max_segment_chars <= 0:
        raise SystemExit("--max-segment-chars must be greater than 0")
    if args.segment_silence_gap_seconds < 0:
        raise SystemExit("--segment-silence-gap-seconds must be 0 or greater")

    asr_model_name = get_asr_model_name(args.asr_model)
    output_path = args.output or default_japanese_srt_path(args.input)
    audio_path = args.input.with_suffix(".wav")

    if audio_path.exists():
        print(f"Using existing audio file: {audio_path}")
    else:
        print(f"Extracting audio from {args.input}...")
        extract_audio(args.input, audio_path)

    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"Using temporary directory: {tmp_dir}")
        print(f"Transcribing with {asr_model_name}...")
        segments = transcribe(
            audio_path,
            Path(tmp_dir),
            asr_model_name,
            chunk_seconds=args.chunk_seconds,
            chunk_overlap_seconds=args.chunk_overlap_seconds,
            max_segment_duration=args.max_segment_duration_seconds,
            max_segment_chars=args.max_segment_chars,
            segment_silence_gap_seconds=args.segment_silence_gap_seconds,
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
