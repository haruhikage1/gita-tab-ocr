# GTRS - Guitar Tablature Recognition System

A deep learning-based OCR system for recognizing guitar tablature from images and converting them to structured formats.

## Features

- **Image Preprocessing**: Automatic deskewing, noise removal, and binarization
- **Staff Detection**: Robust detection of tab staffs in complex layouts
- **Symbol Recognition**:识别品位数字、技巧标记、休止符等
- **Multiple Output Formats**:
  - MusicXML - 标准的音乐符号格式
  - ASCII Tab - 纯文本吉他谱格式
  - JSON - 结构化数据输出

## Supported Guitar Tunings

- Standard (E-A-D-G-B-E)
- Drop D (D-A-D-G-B-E)
- Drop C (C-G-C-F-A-D)
- Open G (D-G-D-G-B-D)
- Open D (D-A-D-F#-A-D)
- Eb (Eb-Ab-Db-Gb-Bb-Eb)

## Installation

```bash
# Clone the repository
git clone https://github.com/haruhikage1/gita-tab-ocr.git
cd gita-tab-ocr

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
# Process a single image
python -m gtrs.main your_tab_image.png

# Process a folder
python -m gtrs.main /path/to/folder/
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--format` | Output format (musicxml/ascii/json) | musicxml |
| `--tuning` | Guitar tuning preset | standard |
| `--output` | Output directory | Same as input |
| `--gpu` | GPU inference mode (no/auto/force) | auto |
| `--debug` | Enable debug output | False |
| `--cache` | Enable caching | False |

### Examples

```bash
# Output as ASCII tab
python -m gtrs.main tab.png --format ascii

# Output as JSON with debug images
python -m gtrs.main tab.png --format json --debug

# Use Drop D tuning
python -m gtrs.main tab.png --tuning drop_d

# Process folder and save to specific directory
python -m gtrs.main /images/tabs/ --output /output/
```

## Project Structure

```
gtrs/
├── api/                  # API server
├── crawler/              # Data crawler for training
├── output/               # Output generators (MusicXML, ASCII, JSON)
├── segmentation/        # Image segmentation models
├── tab_staff_detection/ # Tab staff detection
├── transformer/         # Transformer model for sequence recognition
├── bounding_boxes.py    # Bounding box utilities
├── constants.py        # Configuration constants
├── main.py             # Main entry point
├── model.py            # Model definitions
├── preprocessing.py    # Image preprocessing
├── staff_dewarping.py  # Staff dewarp
└── staff_parsing.py    # Staff parsing logic
```

## Technical Details

- **Input**: PNG, JPG, JPEG images of guitar tablature
- **Max Image Width**: 1000px (auto-resize larger images)
- **Model**: ONNX runtime inference
- **Segmentation Classes**: 8 (tab lines, fret numbers, techniques, clef, rhythm, bar lines, etc.)

## Requirements

- Python 3.8+
- OpenCV
- NumPy
- ONNX Runtime
- See `requirements.txt` for full list

## License

MIT License

## Acknowledgments

Based on the Guitar Tablature Recognition System (GTRS) project.
