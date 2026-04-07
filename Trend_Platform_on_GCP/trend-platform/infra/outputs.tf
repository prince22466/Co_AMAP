# Outputs expose important created resource identifiers after `terraform apply`.
output "dataset_id" {
  value = google_bigquery_dataset.curated.dataset_id
}

output "cluster_name" {
  value = google_container_cluster.autopilot.name
}

output "artifact_repository" {
  value = google_artifact_registry_repository.repo.repository_id
}
