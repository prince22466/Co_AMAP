variable "project_id" {
  description = "GCP project ID"
  type        = string
  default     = "tests101-483015"
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
