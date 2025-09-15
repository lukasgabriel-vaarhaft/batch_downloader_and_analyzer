# Data Labeling Project

A comprehensive image dataset processing pipeline for generating descriptions, categorizing images, and analyzing distributions between real and generated images.

## Overview

This project processes two types of image datasets:
- **REAL**: Real-world images 
- **GEN**: AI-generated images (DALL-E 2, etc.)

The pipeline generates detailed descriptions using Google's Gemini API, categorizes them into predefined categories, and creates comparative visualizations.

## Features

- 🖼️ **Automated Image Description**: Generate detailed JSON-structured descriptions using Gemini API
- 🏷️ **Smart Categorization**: Automatically categorize images into predefined categories
- 📊 **Distribution Analysis**: Create comparative plots showing category distributions
- ☁️ **S3 Integration**: Download images from AWS S3 with robust error handling
- ⚡ **Parallel Processing**: Multi-threaded processing for faster execution
- 📈 **Progress Tracking**: Real-time progress bars and resumable operations

## Project Structure

```
DataLabeling/
├── downloader/                 # S3 download functionality
│   ├── download.py                 # Complete download script with analysis & progress bar
│   └── README.md                   # Downloader documentation
├── scripts/                    # Main processing scripts
│   ├── generate_descriptions.py    # Generate image descriptions
│   ├── categorize_descriptions.py  # Categorize descriptions
│   ├── plot_category_distribution.py        # Single dataset plots
│   ├── plot_combined_distribution.py        # Combined comparison plots
│   ├── plot_normalized_distribution.py     # Normalized comparison plots
│   └── extract_categories.py      # Extract unique categories
├── categorys/                  # Category definitions
│   ├── unique_categories.csv       # List of available categories
│   └── categorize.py              # Category extraction logic
├── csv_real_gen/              # Dataset file lists
│   ├── real.csv                   # List of REAL image filenames
│   └── gen.csv                    # List of GEN image filenames
├── reports/                   # Generated reports and visualizations
│   ├── real_descriptions_categorized.csv   # REAL images with categories
│   ├── gen_descriptions_categorized.csv    # GEN images with categories
│   ├── category_distribution.png           # REAL distribution plot
│   ├── gen_category_distribution.png       # GEN distribution plot
│   ├── combined_category_distribution.png  # Side-by-side comparison
│   └── normalized_category_distribution.png # Normalized comparison
├── requirements.txt           # Python dependencies
├── config.yaml.example       # Configuration template
└── README.md                 # This file
```

## Setup

### Prerequisites

- Python 3.8+
- AWS CLI configured with S3 access (optional, for downloading)
- Google Gemini API key

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd DataLabeling
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API keys**
   ```bash
   cp config.yaml.example config.yaml
   # Edit config.yaml with your Gemini API key
   ```

## Usage

### 1. Download Images (Optional)

If your images are stored on S3:

```bash
# Download REAL images (with confirmation prompt)
python downloader/download.py --csv-file csv_real_gen/real.csv --output-dir REAL --s3-prefix "datalake_training/REAL/"

# Download GEN images (auto-confirm for scripts)
python downloader/download.py --csv-file csv_real_gen/gen.csv --output-dir GEN --s3-prefix "datalake_training/GEN/" --auto-confirm

# Custom download with your own paths
python downloader/download.py --csv-file my_images.csv --output-dir my_images/ --s3-prefix "my-bucket-folder/" --bucket my-bucket
```

**Smart Download Features:**
- 🔍 **Pre-download Analysis** - Scans S3 and shows detailed statistics before starting
- 📊 **Interactive Summary** - Shows available/missing files with confirmation prompt
- 📈 **Progress Bar** - Real-time download progress with current file display
- 🔄 **Resumable Downloads** - Stop (Ctrl+C) and resume anytime
- 💾 **State Management** - Progress saved in `{output-dir}/download_state.json`
- 📝 **Detailed Logs** - Complete logs in `{output-dir}/logs/`
- 🤖 **Auto-confirm** - Use `--auto-confirm` for scripted/automated downloads

**Directory Structure Created:**
```
your-output-dir/
├── download_state.json    # Resume state
├── logs/                  # Download logs
│   └── download_YYYYMMDD_HHMMSS.log
└── [your images]         # Downloaded images
```

### 2. Generate Descriptions

Generate detailed descriptions for your images:

```bash
# For REAL images
python scripts/generate_descriptions.py --images-dir REAL --output-csv data/real_descriptions.csv --num-workers 10

# For GEN images  
python scripts/generate_descriptions.py --images-dir GEN --output-csv data/gen_descriptions.csv --num-workers 10
```

### 3. Categorize Images

Categorize the generated descriptions:

```bash
# For REAL images
python scripts/categorize_descriptions.py --input-csv data/real_descriptions.csv --output-csv reports/real_descriptions_categorized.csv

# For GEN images
python scripts/categorize_descriptions.py --input-csv data/gen_descriptions.csv --output-csv reports/gen_descriptions_categorized.csv
```

### 4. Generate Visualizations

Create distribution plots:

```bash
# Individual distribution plots
python scripts/plot_category_distribution.py --csv-path reports/real_descriptions_categorized.csv --output-path reports/category_distribution.png

# Combined comparison plot
python scripts/plot_combined_distribution.py --real-csv reports/real_descriptions_categorized.csv --gen-csv reports/gen_descriptions_categorized.csv

# Normalized comparison plot
python scripts/plot_normalized_distribution.py --real-csv reports/real_descriptions_categorized.csv --gen-csv reports/gen_descriptions_categorized.csv
```

## Configuration

The `config.yaml` file contains important settings:

```yaml
gemini_api_key: "your-api-key-here"
caption_model: "gemini-2.5-flash-lite"
processing:
  image_max_size: 768
  max_retries: 3
  retry_backoff_seconds: 2.0
  per_request_sleep_seconds: 0.0
```

## Output Formats

### Description CSV Format
```csv
image_name,description
image1.jpg,"A detailed description of the image content..."
```

### Categorized CSV Format
```csv
image_name,description,categories
image1.jpg,"A detailed description...","category1, category2"
```

## Features in Detail

### Robust Processing
- **Resumable Operations**: All scripts can resume from where they left off
- **Error Handling**: Comprehensive error handling with retry logic
- **Progress Tracking**: Real-time progress bars using tqdm
- **Parallel Processing**: Configurable multi-threading for faster processing

### Flexible S3 Integration
- **Extension-Agnostic Lookup**: Matches files regardless of extension (.jpg vs .png)
- **Recursive Search**: Searches through all subdirectories
- **Batch Operations**: Efficient bulk operations with pagination

### Smart Categorization
- **JSON Response Parsing**: Robust parsing of Gemini API responses
- **Category Validation**: Ensures only valid categories are assigned
- **Fallback Handling**: Graceful handling of parsing errors

## Troubleshooting

### Common Issues

1. **API Rate Limits**: Adjust `per_request_sleep_seconds` in config.yaml
2. **Memory Issues**: Reduce `num_workers` parameter
3. **AWS Token Expiry**: Run `aws sso login` to refresh credentials
4. **Missing Categories**: Check `categorys/unique_categories.csv` for available categories

### Logs and Debugging

All scripts provide detailed logging. Enable debug mode by setting the logging level in the scripts.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here]

## Acknowledgments

- Google Gemini API for image description generation
- AWS S3 for image storage
- Python community for excellent libraries (tqdm, pandas, matplotlib, etc.)
