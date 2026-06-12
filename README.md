# Parakeet Subtitle Transcriber & Translator

这是一个基于 Parakeet 日语模型开发的语音转字幕工具，并集成了 Google Gemini API 进行高质量的中文字幕翻译。

## 功能特点

- **高精度识别**：默认使用日语优化的 Parakeet 模型；在 Apple Silicon 上优先走 MLX 后端。
- **自动切片**：支持长视频/音频自动切片处理，避免显存溢出或处理超时。
- **智能合并**：自动合并相邻且内容重叠的字幕片段。
- **Gemini 翻译**：自动将生成的日语字幕翻译成自然、流畅的简体中文。
- **独立工具**：提供独立的命令行工具，可对已有的 SRT 文件进行翻译。

## 安装指南

### 前置要求

- Python >= 3.11
- [FFmpeg](https://ffmpeg.org/) (用于音频提取和处理)
- [uv](https://github.com/astral-sh/uv) (推荐的 Python 包管理工具)

### 快速开始

1. **克隆仓库**:
   ```bash
   git clone https://github.com/mirtu-ko/parakeet-ja-sub.git
   cd Parakeet
   ```

2. **安装依赖**:
   ```bash
   uv sync
   ```

3. **配置 API Key**:
   将项目根目录下的 `.env.example` 重命名为 `.env`，并填入你的 Gemini API Key：
   ```env
   GEMINI_API_KEY=your_actual_api_key_here
   ```

## 使用方法

### 1. 全流程：提取音频 + 语音转写 + 自动翻译

直接对视频文件进行处理：
```bash
uv run parakeet path/to/video.mp4
```
- 执行后将生成 `video.srt` (日语) 和 `video.cn.srt` (中文)。
- 你可以使用 `-o` 指定输出路径，或使用 `--chunk-seconds` 调整切片长度。
- Apple Silicon 默认会使用 `mlx-community/parakeet-tdt_ctc-0.6b-ja`，其他平台默认使用 `nvidia/parakeet-tdt_ctc-0.6b-ja`。

### 2. 仅翻译：对已有字幕进行翻译

如果你已经有了 `.srt` 文件，只想使用 Gemini 进行翻译：
```bash
uv run parakeet-translate path/to/subtitle.srt
```
- 默认生成 `subtitle.cn.srt`。
- 支持使用 `-o` 参数指定输出文件名。

## 参数说明

### `parakeet` (转写+翻译)
- `input`: 输入视频路径（必填）。
- `-o`, `--output`: 输出 SRT 路径（可选）。
- `--chunk-seconds`: 切片时长，默认 20 秒。
- `--chunk-overlap-seconds`: 切片重叠时长，默认 2 秒。
- `--asr-model`: 指定 ASR 模型；也可通过 `PARAKEET_ASR_MODEL` 环境变量覆盖默认值。

### `parakeet-translate` (仅翻译)
- `input`: 输入 SRT 路径（必填）。
- `-o`, `--output`: 输出中文字幕路径（可选）。

## 模型后端

- Apple Silicon: 默认使用 `parakeet-mlx` 加载 `mlx-community/parakeet-tdt_ctc-0.6b-ja`
- 其他平台: 默认使用 NeMo 加载 `nvidia/parakeet-tdt_ctc-0.6b-ja`
- 你也可以手动切换：
  ```bash
  uv run parakeet path/to/video.mp4 --asr-model mlx-community/parakeet-tdt_ctc-0.6b-ja
  ```

## 技术栈

- **ASR**: Parakeet JA (via MLX on Apple Silicon, NeMo elsewhere)
- **Translation**: Google Gemini Flash Lite
- **Tools**: `pysrt`, `nemo_toolkit`, `ffmpeg`, `uv`

## 许可证

MIT
