#!/usr/bin/env python3
"""Build and push API/worker images to Artifact Registry."""
"""using cloud build and submit."""
"""docker build and push doesnt work in GCP."""

import argparse
import subprocess
import sys
from pathlib import Path

def run_command(cmd, cwd=None):
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, text=True)
    if result.returncode != 0:
        print(f"Error: Command failed with exit code {result.returncode}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Build and push Docker images to Google Cloud Artifact Registry.")
    parser.add_argument("--project-id", default="tests101-483015")
    parser.add_argument("--region", default="us-central1")
    parser.add_argument("--repo", default="trend-platform", help="")
    parser.add_argument("--tag", default="latest", help="Image tag")
    
    args = parser.parse_args()

    base_image_path = f"{args.region}-docker.pkg.dev/{args.project_id}/{args.repo}"
    
    images = {
        "api": "app/api",
        "worker": "app/worker"
    }
    
    root_dir = Path(__file__).resolve().parent.parent

    for image_name, build_context in images.items():
        full_image_name = f"{base_image_path}/{image_name}:{args.tag}"
        
        print(f"\n--- Submitting {image_name} to Cloud Build ---")
        run_command(["gcloud", "builds", "submit", "--tag", full_image_name, "--project", args.project_id, build_context], cwd=root_dir)
        
    print("\nAll images built and pushed successfully!")

if __name__ == "__main__":
    main()
