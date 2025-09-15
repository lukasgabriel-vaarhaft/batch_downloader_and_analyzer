# Image Downloader Module

Professional S3 image downloading tool with advanced features for data labeling projects.

## Features

🔍 **Smart Pre-Download Analysis**
- Scans S3 bucket and analyzes file availability
- Shows detailed statistics before starting
- Lists missing files with clear warnings

📊 **Interactive Download Summary**
- Comprehensive statistics display
- User confirmation prompt (can be bypassed with `--auto-confirm`)
- Clear breakdown of available vs missing files

📈 **Real-time Progress Tracking**
- Beautiful progress bar with tqdm
- Current file name display
- Speed and ETA indicators

🔄 **Resumable Downloads**
- Automatic state management with JSON files
- Graceful interruption handling (Ctrl+C)
- Resume from exactly where you left off

💾 **Professional Logging**
- Dual output: console + detailed log files
- Timestamped entries for all operations
- Organized log directory structure

🤖 **Extension-Agnostic Matching**
- Finds files regardless of extension (.jpg vs .png)
- Intelligent filename matching
- Handles S3 subdirectory structures

## Quick Start

```bash
# Basic download with confirmation
python downloader/download.py --csv-file data.csv --output-dir images/ --s3-prefix "my-folder/"

# Automated download for scripts
python downloader/download.py --csv-file data.csv --output-dir images/ --s3-prefix "my-folder/" --auto-confirm

# Custom S3 bucket
python downloader/download.py --csv-file data.csv --output-dir images/ --s3-prefix "folder/" --bucket my-bucket
```

## Directory Structure Created

```
your-output-dir/
├── download_state.json              # Resume state tracking
├── logs/                           # Detailed operation logs
│   └── download_YYYYMMDD_HHMMSS.log
└── [downloaded images]             # Your actual image files
```

## Command Reference

### Required Arguments
- `--csv-file`: Path to CSV file containing image filenames
- `--output-dir`: Local directory to save downloaded images  
- `--s3-prefix`: S3 folder path (e.g., "datalake_training/images/")

### Optional Arguments
- `--bucket`: S3 bucket name (default: "vaarhaft-ml-core")
- `--auto-confirm`: Skip confirmation prompt for automated use
- `--resume`: Resume previous download (enabled by default)

## Example Workflows

### Interactive Download
```bash
python downloader/download.py \
  --csv-file csv_real_gen/real.csv \
  --output-dir REAL_IMAGES \
  --s3-prefix "datalake_training/REAL/"
```

This will:
1. 🔍 Scan S3 and build file map
2. 📊 Show detailed analysis summary
3. ❓ Ask for confirmation to proceed
4. 📥 Download with progress bar
5. 💾 Save state for resumability

### Automated/Scripted Download
```bash
python downloader/download.py \
  --csv-file csv_real_gen/gen.csv \
  --output-dir GEN_IMAGES \
  --s3-prefix "datalake_training/GEN/" \
  --auto-confirm
```

### Resume Interrupted Download
Simply run the same command again - the tool automatically detects and resumes from the last completed file.

## Advanced Features

### State Management
- **Automatic**: Progress saved after each successful download
- **Resumable**: Restart with identical command to continue
- **Robust**: Handles network interruptions gracefully

### Error Handling
- **AWS Token Expiry**: Clear instructions for renewal
- **Missing Files**: Tracked separately, don't block other downloads
- **Network Issues**: Logged with retry recommendations

### Performance
- **Efficient S3 Scanning**: Uses pagination for large buckets
- **Smart Caching**: Builds filename map once, reuses for all lookups
- **Optimized I/O**: Minimal disk operations, maximum throughput

## Troubleshooting

### Common Issues

**AWS Token Expired**
```
❌ AWS token expired. Run: aws sso login
```
Solution: Run `aws sso login` and restart the download.

**Files Not Found**
The tool will show exactly which files are missing and continue with available ones.

**Interrupted Download**
Just run the same command again - it will resume automatically.

### Log Analysis
Check `{output-dir}/logs/download_YYYYMMDD_HHMMSS.log` for detailed operation logs including:
- Individual file download status
- Error details and timestamps  
- Performance metrics
- State changes

## Integration

This downloader integrates seamlessly with the main data labeling pipeline:

1. **Download** images using this tool
2. **Generate descriptions** with `scripts/generate_descriptions.py`
3. **Categorize** with `scripts/categorize_descriptions.py`
4. **Analyze** with visualization scripts

## Files in this Directory

- `download.py` - Complete download script with all features
- `README.md` - This documentation
