# Batch Downloader and Analyzer

A professional-grade image dataset processing pipeline for downloading, describing, categorizing, and analyzing large-scale image datasets from S3. Perfect for machine learning research, data labeling projects, and comparative analysis of real vs. AI-generated content.

## 🚀 Key Features

- **🔍 Smart S3 Downloader** - Professional downloader with resume functionality, progress bars, and intelligent analysis
- **🤖 AI-Powered Descriptions** - Generate detailed image descriptions using Google's Gemini API
- **🏷️ Automated Categorization** - Classify images into predefined categories with high accuracy
- **📊 Advanced Analytics** - Create comprehensive visualizations and comparative analysis
- **⚡ Production-Ready** - Robust error handling, logging, and state management
- **🔄 Resumable Operations** - All processes can be interrupted and resumed seamlessly

## 📋 Use Cases

- **Machine Learning Research** - Process large datasets for training and evaluation
- **Data Quality Analysis** - Compare real vs. AI-generated image distributions  
- **Content Categorization** - Automatically organize and label image collections
- **Dataset Migration** - Efficiently download and process images from cloud storage
- **Comparative Studies** - Analyze differences between different image generation models

## 🏗️ Architecture

The pipeline consists of four main components:

1. **📥 Downloader Module** (`downloader/`) - Professional S3 downloading with analysis
2. **🔧 Processing Scripts** (`scripts/`) - Core image processing and analysis tools  
3. **📊 Visualization Tools** - Generate comprehensive charts and comparisons
4. **⚙️ Configuration System** - Flexible YAML-based configuration management

## Project Structure

```
batch_downloader_and_analyzer/
├── downloader/                 # 📥 S3 Download Module
│   ├── batch_download.py           # Smart batch downloader with auto-configuration
│   ├── download.py                 # Manual downloader with full control
│   ├── to_download/                # Drop CSV files here for auto-processing
│   │   ├── GEN_example.csv         # Example GEN dataset file list
│   │   └── REAL_example.csv        # Example REAL dataset file list
│   └── README.md                   # Detailed downloader documentation
├── scripts/                    # 🔧 Core Processing Scripts
│   ├── generate_descriptions.py    # AI-powered image description generation
│   ├── categorize_descriptions.py  # Automated image categorization
│   ├── plot_category_distribution.py        # Single dataset visualization
│   ├── plot_combined_distribution.py        # Comparative analysis plots
│   ├── plot_normalized_distribution.py     # Normalized comparison charts
│   ├── extract_categories.py      # Category extraction utilities
│   └── main.py                     # Legacy processing pipeline
├── requirements.txt           # Python dependencies
├── config.yaml.example       # Configuration template
├── .gitignore                # Git exclusion rules
└── README.md                 # This documentation

# Data directories (created during usage, excluded from git)
├── data/                     # Generated descriptions and processed data
├── reports/                  # Analysis results and visualizations  
├── csv_real_gen/            # Dataset file lists
├── REAL/                    # Real image dataset
└── GEN/                     # Generated image dataset
```

## ⚡ Quick Start

### 🚀 Try the Examples

Get started immediately with the included example datasets:

```bash
# Clone and setup
git clone git@github.com:lukasgabriel-vaarhaft/batch_downloader_and_analyzer.git
cd batch_downloader_and_analyzer
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Copy config template and add your Gemini API key
cp config.yaml.example config.yaml
# Edit config.yaml with your API key

# Run the interactive batch downloader
cd downloader
python batch_download.py

# Select files to download and watch the magic happen!
```

## 🛠️ Detailed Setup

### Prerequisites

- **Python 3.8+** - Modern Python installation
- **AWS CLI** - Configured with S3 access (for downloading images)
- **Google Gemini API Key** - For AI-powered image analysis

### Installation

1. **Clone the repository**
   ```bash
   git clone git@github.com:lukasgabriel-vaarhaft/batch_downloader_and_analyzer.git
   cd batch_downloader_and_analyzer
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

5. **Setup AWS credentials (for S3 downloading)**
   ```bash
   aws configure
   # or for SSO:
   aws sso login
   ```

## 🚀 Usage Guide

### Step 1: Download Images from S3

#### 🎯 Smart Batch Download (Recommended)

The easiest way to get started - just drop your CSV files and let the system auto-configure everything:

```bash
# Interactive batch download with file selection
cd downloader
python batch_download.py

# This will:
# 1. 🔍 Find all CSV files in to_download/ folder
# 2. 🎛️  Let you select which ones to process
# 3. 🤖 Auto-configure S3 paths based on filenames
# 4. 📊 Show detailed analysis for each dataset
# 5. 📥 Download with progress tracking
```

**Auto-Configuration Examples:**
- `GEN_experiment.csv` → Downloads to `downloads/GEN_experiment_downloads/`
- `REAL_validation.csv` → Downloads to `downloads/REAL_validation_downloads/`

#### 🔧 Manual Download (Advanced)

For full control over paths and configuration:

```bash
# Manual configuration
python downloader/download.py \
  --csv-file your_image_list.csv \
  --output-dir LOCAL_IMAGES \
  --s3-prefix "your-s3-folder/"
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

### Step 2: Generate AI Descriptions

Create detailed descriptions using Google's Gemini API:

```bash
# Process your downloaded images
python scripts/generate_descriptions.py \
  --dataset real \
  --num-workers 10

# The script automatically:
# - Finds images in REAL/ directory
# - Generates JSON-structured descriptions
# - Saves to data/real_descriptions.csv
# - Supports resume functionality
```

### Step 3: Categorize Images

Automatically categorize images based on their descriptions:

```bash
# Categorize your processed images
python scripts/categorize_descriptions.py \
  --input-csv data/real_descriptions.csv \
  --output-csv reports/real_descriptions_categorized.csv \
  --categories-csv categorys/unique_categories.csv
```

### Step 4: Generate Analysis & Visualizations

Create comprehensive visualizations and comparisons:

```bash
# Single dataset distribution
python scripts/plot_category_distribution.py \
  --csv-path reports/real_descriptions_categorized.csv \
  --output-path reports/distribution.png

# Compare two datasets
python scripts/plot_combined_distribution.py \
  --real-csv reports/real_descriptions_categorized.csv \
  --gen-csv reports/gen_descriptions_categorized.csv

# Normalized comparison (percentages)
python scripts/plot_normalized_distribution.py \
  --real-csv reports/real_descriptions_categorized.csv \
  --gen-csv reports/gen_descriptions_categorized.csv
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

## 📈 Performance & Scalability

- **Multi-threaded Processing** - Configurable worker count for optimal performance
- **Memory Efficient** - Processes large datasets without memory issues
- **Resumable Operations** - Handle interruptions gracefully
- **Batch Processing** - Optimized for large-scale operations
- **Rate Limiting** - Respects API limits with configurable delays

## 🔧 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| **API Rate Limits** | Adjust `per_request_sleep_seconds` in config.yaml |
| **Memory Issues** | Reduce `num_workers` parameter |
| **AWS Token Expiry** | Run `aws sso login` to refresh credentials |
| **Missing Categories** | Check category files in your data directory |
| **Network Timeouts** | Increase `max_retries` in configuration |

### Debugging

- **Download Issues**: Check logs in `{output-dir}/logs/`
- **API Errors**: Enable debug logging in scripts
- **State Problems**: Delete `download_state.json` to reset
- **Performance**: Monitor worker count vs. system resources

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Add tests if applicable
5. Commit: `git commit -m 'Add amazing feature'`
6. Push: `git push origin feature/amazing-feature`
7. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Google Gemini API** - For powerful image analysis capabilities
- **AWS S3** - For reliable cloud storage integration
- **Python Community** - For excellent libraries (tqdm, pandas, matplotlib, boto3)
- **Open Source Contributors** - For making this project possible

