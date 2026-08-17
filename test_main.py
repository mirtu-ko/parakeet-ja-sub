import unittest
from pathlib import Path
from sys import modules
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

from main import (
    SubtitleSegment,
    filter_chars_to_window,
    get_hypothesis_timestamps,
    group_chars_into_segments,
    merge_adjacent_segments,
    normalize_segments,
    split_audio,
    transcribe_with_mlx,
    translate_batch_with_fallback,
)


class GetHypothesisTimestampsTests(unittest.TestCase):
    def test_reads_nemo_timestep_field(self):
        timestep = {"char": [{"char": "a", "start": 1.0, "end": 1.1}]}
        hypothesis = SimpleNamespace(timestep=timestep, timestamp=[])

        self.assertIs(get_hypothesis_timestamps(hypothesis), timestep)

    def test_supports_timestamp_field_from_other_model_versions(self):
        timestamp = {"segment": [{"segment": "text", "start": 1.0, "end": 2.0}]}
        hypothesis = SimpleNamespace(timestamp=timestamp)

        self.assertIs(get_hypothesis_timestamps(hypothesis), timestamp)

    def test_skips_empty_timestep_when_timestamp_has_data(self):
        timestep = {"char": []}
        timestamp = {"segment": [{"segment": "text", "start": 1.0, "end": 2.0}]}
        hypothesis = SimpleNamespace(timestep=timestep, timestamp=timestamp)

        self.assertIs(get_hypothesis_timestamps(hypothesis), timestamp)

    def test_returns_empty_dict_without_decoded_timestamps(self):
        hypothesis = SimpleNamespace(timestamp=[])

        self.assertEqual(get_hypothesis_timestamps(hypothesis), {})


class GroupCharsIntoSegmentsTests(unittest.TestCase):
    def test_ignores_empty_decoder_token_before_first_spoken_character(self):
        chars = [
            {"char": [""], "start": 0.0, "end": 0.32},
            {"char": ["予"], "start": 13.44, "end": 13.68},
            {"char": ["備"], "start": 13.68, "end": 13.92},
        ]

        segments = group_chars_into_segments(chars, time_offset=18.0)

        self.assertEqual(len(segments), 1)
        self.assertAlmostEqual(segments[0]["start"], 31.44)
        self.assertAlmostEqual(segments[0]["end"], 31.92)
        self.assertEqual(segments[0]["text"], "予備")


class SegmentNormalizationTests(unittest.TestCase):
    def test_attaches_delayed_punctuation_without_extending_subtitle_duration(self):
        segments: list[SubtitleSegment] = [
            {"start": 2716.56, "end": 2718.48, "text": "またいっぱい気持ちよくなろう"},
            {"start": 2723.2, "end": 2723.52, "text": "。"},
        ]

        self.assertEqual(
            merge_adjacent_segments(segments),
            [{"start": 2716.56, "end": 2718.48, "text": "またいっぱい気持ちよくなろう。"}],
        )

    def test_does_not_merge_repeated_text_across_long_gap(self):
        segments: list[SubtitleSegment] = [
            {"start": 1.0, "end": 1.5, "text": "はい"},
            {"start": 60.0, "end": 60.5, "text": "はい"},
        ]

        self.assertEqual(merge_adjacent_segments(segments), segments)

    def test_merges_duplicate_text_from_overlapping_chunks(self):
        segments: list[SubtitleSegment] = [
            {"start": 18.0, "end": 20.0, "text": "はい"},
            {"start": 18.2, "end": 20.5, "text": "はい"},
        ]

        self.assertEqual(
            merge_adjacent_segments(segments),
            [{"start": 18.0, "end": 20.5, "text": "はい"}],
        )

    def test_caps_duration_when_text_is_too_short_to_split(self):
        segments: list[SubtitleSegment] = normalize_segments(
            [{"start": 0.0, "end": 60.0, "text": "はい"}],
            max_segment_duration=20.0,
            max_segment_chars=45,
        )

        self.assertEqual(segments, [{"start": 0.0, "end": 20.0, "text": "はい"}])


class ChunkOwnershipTests(unittest.TestCase):
    @patch("main.get_audio_duration", return_value=40.0)
    @patch("main.run_command")
    def test_overlap_is_owned_by_the_following_chunk(self, _run_command, _get_duration):
        with TemporaryDirectory() as tmp_dir:
            chunks = split_audio(Path("audio.wav"), Path(tmp_dir), 20.0, 2.0)

        self.assertEqual(
            [(chunk.keep_start, chunk.keep_end) for chunk in chunks],
            [(0.0, 18.0), (18.0, 36.0), (36.0, 40.0)],
        )

    def test_each_character_belongs_to_only_one_window(self):
        chars = [
            {"char": ["前"], "start": 0.8, "end": 1.0},
            {"char": ["後"], "start": 1.0, "end": 1.2},
        ]

        left = filter_chars_to_window(chars, time_offset=17.0, keep_start=0.0, keep_end=18.0)
        right = filter_chars_to_window(chars, time_offset=17.0, keep_start=18.0, keep_end=36.0)

        self.assertEqual(left, [chars[0]])
        self.assertEqual(right, [chars[1]])


class TranslationFallbackTests(unittest.TestCase):
    @patch("main.translate_batch", side_effect=ValueError("different shape"))
    def test_single_line_failure_is_not_silently_kept_as_japanese(self, _translate_batch):
        with self.assertRaisesRegex(RuntimeError, "could not translate"):
            translate_batch_with_fallback(
                cast(Any, object()),
                "model",
                ["原文"],
                None,
                0.0,
                0,
            )


class MlxTranscriptionTests(unittest.TestCase):
    def test_passes_sentence_timing_limits_to_mlx(self):
        captured: dict[str, object] = {}

        class SentenceConfig:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class DecodingConfig:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class FakeModel:
            def transcribe(self, _audio_path, **kwargs):
                captured.update(kwargs)
                return SimpleNamespace(sentences=[])

        fake_module = ModuleType("parakeet_mlx")
        setattr(fake_module, "DecodingConfig", DecodingConfig)
        setattr(fake_module, "SentenceConfig", SentenceConfig)
        setattr(fake_module, "from_pretrained", lambda _model_name: FakeModel())

        with patch.dict(modules, {"parakeet_mlx": fake_module}), patch("main.is_apple_silicon", return_value=True):
            transcribe_with_mlx(Path("audio.wav"), "mlx-community/model", 20.0, 2.0, 8.0, 45, 0.8)

        decoding_config = cast(Any, captured["decoding_config"])
        self.assertEqual(decoding_config.sentence.silence_gap, 0.8)
        self.assertEqual(decoding_config.sentence.max_duration, 8.0)


if __name__ == "__main__":
    unittest.main()
