#!/usr/bin/env python3
"""Deploy Kubernetes manifests to GKE."""

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
    parser = argparse.ArgumentParser(description="Deploy Docker images to GKE.")
    parser.add_argument("--project-id", default="tests101-483015")
    parser.add_argument("--region", default="us-central1")
    parser.add_argument("--cluster-name", default="trend-platform-autopilot", help="Name of the GKE cluster")
    parser.add_argument("--cluster-location", default="us-central1", help="Zone or region of the GKE cluster")
    parser.add_argument("--repo", default="trend-platform", help="Artifact Registry repository")
    parser.add_argument("--tag", default="latest", help="Image tag to deploy")
    parser.add_argument("--k8s-dir", default="k8s", help="Directory containing Kubernetes manifests")
    
    args = parser.parse_args()

    base_image_path = f"{args.region}-docker.pkg.dev/{args.project_id}/{args.repo}"
    root_dir = Path(__file__).resolve().parent.parent
    k8s_dir = root_dir / args.k8s_dir

    print("\n--- Authenticating with GKE ---")
    run_command([
        "gcloud", "container", "clusters", "get-credentials",
        args.cluster_name,
        "--location", args.cluster_location,
        "--project", args.project_id
    ])

    print("\n--- Applying Kubernetes Manifests ---")
    if k8s_dir.exists():
        namespace_file = k8s_dir / "namespace.yaml"
        namespace = "default"
        
        if namespace_file.exists():
            print("Applying namespace first...")
            run_command(["kubectl", "apply", "-f", str(namespace_file)])
            
            # Extract namespace name from the yaml file
            with open(namespace_file, "r") as f:
                for line in f:
                    if line.strip().startswith("name:"):
                        namespace = line.split(":", 1)[1].strip()
                        break

        cmd = ["kubectl", "delete", "jobs", "--all", "-n", namespace, "--ignore-not-found"]
        subprocess.run(cmd, check=True)
        print("ALL running Jobs deleted successfully.")

        run_command(["kubectl", "apply", "-f", str(k8s_dir)])
    else:
        print(f"Warning: Kubernetes directory not found at {k8s_dir}. Make sure you have your YAML manifests there.")

    print("\nDeployment process completed! Run 'kubectl get pods' to verify.")

if __name__ == "__main__":
    main()
