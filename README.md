# ChandraOCR Batch Inference

Efficient batched OCR inference using [ChandraOCR](https://huggingface.co/datalab-to/chandra) on Apple Silicon via MLX.

This repository provides a streamlined pipeline for running ChandraOCR on large collections of images with optimized batch processing. It includes a modified version of [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) (v0.3.10) with some changes to support batch generation for Chandra.

## Features

- **Batch Processing**: Process multiple images efficiently with automatic grouping by image dimensions
- **Apple Silicon Optimized**: Leverages MLX for native Metal acceleration on M1/M2/M3 chips
- **Flexible Input**: Process images from a directory (recursive) or a manifest file
- **Resume Capability**: Resume interrupted processing by skipping already-processed images
- **Dual Output**: Results saved to both CSV (for analysis) and individual text files (per image)
- **HTML Output**: ChandraOCR outputs structured HTML with support for tables, math, and formatting

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- ~16GB+ RAM recommended for bf16 model

## Installation

1. Clone this repository:
```bash
git clone https://github.com/neurodiscoveryai/ndai-mlx-chandraocr.git
cd chandraocr-batch
```

2. Create a virtual environment and install dependencies:
```bash
conda create -n chandraocr_mlx python=3.11.14
conda activate chandraocr_mlx
pip install -r requirements.txt
```

3. Install the modified mlx-vlm package:
```bash
pip install -e mlx-vlm-0.3.10/
```

## Usage

### Basic Usage

Process all images in a directory:
```bash
python ocr_batch.py --images-dir /path/to/images --output results.csv
```

### From a Manifest File

Create a text file with one image path per line, then:
```bash
python ocr_batch.py --manifest image_paths.txt --output results.csv
```

### Advanced Options

```bash
python ocr_batch.py \
    --images-dir /path/to/images \
    --output results.csv \
    --batch-size 8 \
    --max-tokens 8192 \
    --resume \
    --verbose
```

#### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--images-dir` | Directory to scan for images (recursive) | - |
| `--manifest` | File with one image path per line | - |
| `--output`, `-o` | Output CSV path | Required |
| `--batch-size` | Images per batch | 8 |
| `--max-tokens` | Maximum tokens to generate per image | 8192 |
| `--max-size` | Resize longest side to N pixels | 1024 |
| `--resume` | Skip images already in output CSV | False |
| `--verbose` | Print per-batch statistics | False |

### Supported Image Formats

- PNG, JPG/JPEG
- TIFF/TIF
- BMP, WebP

## Output

### CSV Output
The output CSV contains three columns:
- `image_path`: Full path to the source image
- `status`: Processing status (`ok`, `error:empty`, or error details)
- `ocr_text`: Extracted text

### Text Files
Each processed image also generates a corresponding `*_ocr.txt` file with the OCR output.

## Model

This project uses [ChandraOCR](https://huggingface.co/mlx-community/chandra-bf16), a vision-language model optimized for document OCR. The model:

- Outputs structured HTML with proper formatting
- Supports tables with `colspan`/`rowspan` attributes
- Handles mathematical expressions with KaTeX-compatible LaTeX
- Preserves document structure (headers, paragraphs, lists)

## Modifications to mlx-vlm

This repository includes mlx-vlm v0.3.10 with the following additions:

1. **`batch_generate()` function**: Enables efficient batch inference for vision-language models
2. **`group_images_by_shape()` utility**: Groups images by dimensions to avoid padding waste
3. **`BatchGenerator` class**: Manages batch prefill and completion for optimal throughput

These changes enable processing multiple images in a single forward pass, significantly improving throughput compared to sequential processing.

## Performance Tips

1. **Batch Size**: Larger batches improve throughput but require more memory. Start with 8 and adjust based on your hardware.
2. **Image Grouping**: The script automatically groups images by dimensions. Having many same-sized images improves efficiency.
3. **Max Size**: Use `--max-size` to resize large images if you're running low on memory.

## License

This project is licensed under the MIT License. The included mlx-vlm modifications are based on [Blaizzy/mlx-vlm](https://github.com/Blaizzy/mlx-vlm) which is also MIT licensed.

## Acknowledgments

- [ChandraOCR](https://huggingface.co/datalab-to/chandra) - The OCR model and its [MLX-version](https://huggingface.co/mlx-community/chandra-bf16)
- [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) - Vision-language model framework for MLX
- [Apple MLX](https://github.com/ml-explore/mlx) - Machine learning framework for Apple Silicon
