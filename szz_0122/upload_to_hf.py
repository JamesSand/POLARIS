#!/usr/bin/env python3
"""
Upload a local model folder to Hugging Face Hub.

Usage:
    python upload_to_hf.py --local_folder /path/to/model --hf_username YourUsername [--private]
"""

import os
import argparse
from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser(description="Upload model to Hugging Face Hub")
    parser.add_argument(
        "--local_folder",
        type=str,
        required=True,
        help="Path to local model folder to upload"
    )
    parser.add_argument(
        "--hf_username",
        type=str,
        default="JameSand",
        help="Hugging Face username (default: JameSand)"
    )
    parser.add_argument(
        "--private",
        type=str,
        default="False",
        choices=["True", "False"],
        help="Whether to make the repo private (default: False)"
    )
    parser.add_argument(
        "--repo_name",
        type=str,
        default=None,
        help="Custom repository name (default: basename of local_folder)"
    )
    
    args = parser.parse_args()
    
    # Get HF token from environment
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError(
            "HF_TOKEN environment variable not set. "
            "Get your token from https://huggingface.co/settings/tokens"
        )
    
    # Validate local folder
    if not os.path.exists(args.local_folder):
        raise ValueError(f"Local folder does not exist: {args.local_folder}")
    
    # Determine repo name
    if args.repo_name:
        basename = args.repo_name
    else:
        basename = os.path.basename(args.local_folder)
    
    # Convert private flag
    is_private = args.private == "True"
    
    # Create API instance
    api = HfApi(token=hf_token)
    
    # Create repo_id
    repo_id = f"{args.hf_username}/{basename}"
    
    print(f"Uploading model to Hugging Face Hub")
    print(f"  Local folder: {args.local_folder}")
    print(f"  Repository:   {repo_id}")
    print(f"  Private:      {is_private}")
    print()
    
    # Create repository (or reuse existing)
    print("Creating/checking repository...")
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=is_private,
        exist_ok=True  # Skip if already exists
    )
    print(f"✓ Repository ready: https://huggingface.co/{repo_id}")
    print()
    
    # Upload folder
    print("Uploading files...")
    api.upload_folder(
        folder_path=args.local_folder,
        repo_id=repo_id,
        repo_type="model",
    )
    
    print()
    print("=" * 60)
    print("Upload completed successfully!")
    print(f"View your model at: https://huggingface.co/{repo_id}")
    print("=" * 60)


if __name__ == "__main__":
    main()
