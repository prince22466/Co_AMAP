variable "project_id" {
  description = "GCP project ID"
  type        = string
  default     = "ur_project_ID"
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "dataset_id" {
  description = "Bigquery dateset ID"
  type        = string
  default     = "trend_curated"
}

variable "artifact_repo" {
  description = "Artifact Registry repository name"
  type        = string
  default     = "trend-platform"
}

variable "cluster_name" {
  description = "GKE Autopilot cluster name"
  type        = string
  default     = "trend-platform-autopilot"
}

variable "public_dataset_region" {
  description = "the region of the publice bigquery dataset to use"
  type        = string
  default     = "US"
}
