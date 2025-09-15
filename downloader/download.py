#!/usr/bin/env python3
"""
Download images from S3 based on CSV file lists.
Supports resumable downloads with state management and logging.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import List, Set, Tuple

import boto3
from botocore.exceptions import ClientError
from tqdm import tqdm


class DownloadState:
    """Manages download state for resumable downloads."""
    
    def __init__(self, state_file: str):
        self.state_file = state_file
        self.downloaded: Set[str] = set()
        self.failed: Set[str] = set()
        self.skipped: Set[str] = set()
        self.start_time = None
        self.load_state()
    
    def load_state(self):
        """Load existing download state."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    self.downloaded = set(data.get('downloaded', []))
                    self.failed = set(data.get('failed', []))
                    self.skipped = set(data.get('skipped', []))
                    self.start_time = data.get('start_time')
                logging.info(f"Loaded state: {len(self.downloaded)} downloaded, {len(self.failed)} failed, {len(self.skipped)} skipped")
            except Exception as e:
                logging.warning(f"Could not load state file: {e}")
    
    def save_state(self):
        """Save current download state."""
        try:
            data = {
                'downloaded': list(self.downloaded),
                'failed': list(self.failed),
                'skipped': list(self.skipped),
                'start_time': self.start_time,
                'last_update': datetime.now().isoformat()
            }
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logging.error(f"Could not save state: {e}")
    
    def mark_downloaded(self, filename: str):
        """Mark file as successfully downloaded."""
        self.downloaded.add(filename)
        self.failed.discard(filename)  # Remove from failed if it was there
        self.save_state()
    
    def mark_failed(self, filename: str):
        """Mark file as failed to download."""
        self.failed.add(filename)
        self.save_state()
    
    def mark_skipped(self, filename: str):
        """Mark file as skipped (not found on S3)."""
        self.skipped.add(filename)
        self.save_state()
    
    def is_completed(self, filename: str) -> bool:
        """Check if file was already processed."""
        return filename in self.downloaded or filename in self.skipped


def setup_logging(output_dir: str):
    """Setup logging to both file and console."""
    logs_dir = os.path.join(output_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(logs_dir, f'download_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.info(f"Logging to: {log_file}")
    return log_file


def build_s3_key_map(s3_client, bucket: str, prefix: str):
    """Builds a map of filename -> full S3 key for all objects under a prefix."""
    logging.info(f"Building S3 file map for prefix '{prefix}'...")
    key_map = {}
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
    
    for page in pages:
        for obj in page.get('Contents', []):
            filename = os.path.basename(obj['Key'])
            if filename:
                # Store both with and without extension for flexible lookup
                filename_no_ext = os.path.splitext(filename)[0]
                key_map[filename] = obj['Key']
                key_map[filename_no_ext] = obj['Key']
    
    logging.info(f"Found {len(key_map)} files on S3.")
    return key_map


def analyze_download_requirements(csv_path: str, s3_key_map: dict, state: DownloadState) -> Tuple[List[str], List[str], List[str]]:
    """
    Analyzes what files need to be downloaded and provides detailed statistics.
    Returns: (files_to_download, files_available_on_s3, files_missing_from_s3)
    """
    try:
        with open(csv_path, 'r') as f:
            all_files = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        logging.error(f"CSV file not found at {csv_path}")
        return [], [], []
    
    files_to_download = []
    files_available_on_s3 = []
    files_missing_from_s3 = []
    
    print("\n🔍 Analyzing download requirements...")
    
    for filename in tqdm(all_files, desc="Scanning files", unit="files"):
        if state.is_completed(filename):
            continue  # Skip already processed files
            
        filename_no_ext = os.path.splitext(filename)[0]
        s3_key = s3_key_map.get(filename) or s3_key_map.get(filename_no_ext)
        
        if s3_key:
            files_to_download.append(filename)
            files_available_on_s3.append(filename)
        else:
            files_missing_from_s3.append(filename)
    
    return files_to_download, files_available_on_s3, files_missing_from_s3


def display_download_summary(csv_path: str, files_to_download: List[str], files_available_on_s3: List[str], 
                           files_missing_from_s3: List[str], state: DownloadState, auto_confirm: bool = False) -> bool:
    """
    Displays a comprehensive download summary and asks for user confirmation.
    Returns True if user wants to continue, False otherwise.
    """
    try:
        with open(csv_path, 'r') as f:
            total_files_in_csv = len([line.strip() for line in f if line.strip()])
    except FileNotFoundError:
        total_files_in_csv = 0
    
    already_downloaded = len(state.downloaded)
    already_skipped = len(state.skipped)
    
    print("\n" + "="*60)
    print("📊 DOWNLOAD ANALYSIS SUMMARY")
    print("="*60)
    print(f"📋 Total files in CSV:           {total_files_in_csv:,}")
    print(f"✅ Already downloaded:           {already_downloaded:,}")
    print(f"⚠️  Already skipped (not found):  {already_skipped:,}")
    print(f"🔍 Available on S3:              {len(files_available_on_s3):,}")
    print(f"❌ Missing from S3:              {len(files_missing_from_s3):,}")
    print(f"📥 Ready to download:            {len(files_to_download):,}")
    print("="*60)
    
    if len(files_missing_from_s3) > 0:
        print(f"\n⚠️  WARNING: {len(files_missing_from_s3)} files are missing from S3 and will be skipped.")
        if len(files_missing_from_s3) <= 10:
            print("Missing files:")
            for filename in files_missing_from_s3[:10]:
                print(f"   • {filename}")
        else:
            print("First 10 missing files:")
            for filename in files_missing_from_s3[:10]:
                print(f"   • {filename}")
            print(f"   ... and {len(files_missing_from_s3) - 10} more")
    
    if len(files_to_download) == 0:
        print("\n✅ All available files have already been downloaded!")
        return False
    
    print(f"\n🚀 Ready to download {len(files_to_download):,} files.")
    
    if auto_confirm:
        print("Auto-confirm enabled. Starting download...")
        return True
    
    while True:
        response = input("\nDo you want to continue with the download? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            print("Download cancelled by user.")
            return False
        else:
            print("Please enter 'y' for yes or 'n' for no.")


def get_files_to_download(csv_path: str, state: DownloadState) -> List[str]:
    """Gets files that need to be downloaded based on CSV and current state."""
    try:
        with open(csv_path, 'r') as f:
            all_files = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        logging.error(f"CSV file not found at {csv_path}")
        return []
    
    # Filter out already completed files
    remaining_files = [f for f in all_files if not state.is_completed(f)]
    logging.info(f"Total files in CSV: {len(all_files)}")
    logging.info(f"Remaining to download: {len(remaining_files)}")
    
    return remaining_files


def download_file(s3_client, bucket: str, s3_key: str, local_path: str, state: DownloadState, filename: str) -> bool:
    """Downloads a single file from S3 with state tracking."""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Download file
        s3_client.download_file(bucket, s3_key, local_path)
        
        # Mark as completed
        state.mark_downloaded(filename)
        return True
        
    except ClientError as e:
        if e.response['Error']['Code'] == 'ExpiredToken':
            logging.error("AWS token expired. Run: aws sso login")
            return False  # Stop download process
        elif e.response['Error']['Code'] == 'NoSuchKey':
            logging.warning(f"File not found on S3: {filename}")
            state.mark_skipped(filename)
            return True  # Continue with other files
        else:
            logging.error(f"S3 error for {filename}: {e}")
            state.mark_failed(filename)
            return True
    except Exception as e:
        logging.error(f"Download error for {filename}: {e}")
        state.mark_failed(filename)
        return True


def main():
    parser = argparse.ArgumentParser(description="Download images from S3 with resumable functionality")
    parser.add_argument("--csv-file", required=True,
                        help="Path to CSV file with image filenames")
    parser.add_argument("--output-dir", required=True,
                        help="Local directory to save downloaded images")
    parser.add_argument("--s3-prefix", required=True,
                        help="S3 prefix/folder path (e.g., 'datalake_training/REAL/')")
    parser.add_argument("--bucket", default="vaarhaft-ml-core",
                        help="S3 bucket name (default: vaarhaft-ml-core)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume previous download (default: true)")
    parser.add_argument("--auto-confirm", action="store_true",
                        help="Skip confirmation prompt and start download automatically")
    args = parser.parse_args()
    
    # Setup directory structure
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    log_file = setup_logging(args.output_dir)
    
    # Initialize download state
    state_file = os.path.join(args.output_dir, 'download_state.json')
    state = DownloadState(state_file)
    
    if not state.start_time:
        state.start_time = datetime.now().isoformat()
        state.save_state()
    
    logging.info("🚀 Starting download process")
    logging.info(f"📋 Input CSV: {args.csv_file}")
    logging.info(f"📁 Output directory: {args.output_dir}")
    logging.info(f"☁️  S3 path: s3://{args.bucket}/{args.s3_prefix}")
    logging.info(f"📄 State file: {state_file}")
    logging.info(f"📝 Log file: {log_file}")
    
    # Initialize S3 client
    try:
        s3_client = boto3.client('s3')
        s3_key_map = build_s3_key_map(s3_client, args.bucket, args.s3_prefix)
    except Exception as e:
        logging.error(f"Failed to connect to S3: {e}")
        return 1
    
    # Analyze download requirements
    files_to_download, files_available_on_s3, files_missing_from_s3 = analyze_download_requirements(
        args.csv_file, s3_key_map, state
    )
    
    # Display summary and get user confirmation
    if not display_download_summary(args.csv_file, files_to_download, files_available_on_s3, files_missing_from_s3, state, args.auto_confirm):
        return 0
    
    # Mark missing files as skipped in state
    for filename in files_missing_from_s3:
        state.mark_skipped(filename)
    
    logging.info(f"📥 Starting download of {len(files_to_download)} files...")
    
    # Download files with progress bar
    try:
        with tqdm(total=len(files_to_download), desc="Downloading", unit="files") as pbar:
            for filename in files_to_download:
                filename_no_ext = os.path.splitext(filename)[0]
                
                # Try to find the file on S3 (with or without extension)
                s3_key = s3_key_map.get(filename) or s3_key_map.get(filename_no_ext)
                
                if not s3_key:
                    logging.warning(f"File not found on S3: {filename}")
                    state.mark_skipped(filename)
                    pbar.update(1)
                    continue
                
                local_path = os.path.join(args.output_dir, filename)
                
                # Download file
                if not download_file(s3_client, args.bucket, s3_key, local_path, state, filename):
                    # Token expired or critical error, exit gracefully
                    logging.error("Download stopped due to critical error")
                    break
                
                pbar.update(1)
                pbar.set_postfix({"Current": filename[:30] + "..." if len(filename) > 30 else filename})
    
    except KeyboardInterrupt:
        logging.info("Download interrupted by user. Progress saved.")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
    
    # Final summary
    logging.info("\n📊 Download Summary:")
    logging.info(f"✅ Successfully downloaded: {len(state.downloaded)}")
    logging.info(f"⚠️  Skipped (not found on S3): {len(state.skipped)}")
    logging.info(f"❌ Failed: {len(state.failed)}")
    logging.info(f"📁 Files saved to: {args.output_dir}/")
    logging.info(f"📝 Detailed logs: {log_file}")
    logging.info(f"📄 State file: {state_file}")
    
    if state.failed:
        logging.info("💡 Tip: Run the same command again to retry failed downloads")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
