resource "google_project_service" "enabled_apis" {
  for_each = toset([
    "container.googleapis.com",
    "iam.googleapis.com",
    "artifactregistry.googleapis.com",
    "cloudresourcemanager.googleapis.com",
  ])

  project            = var.project_id
  service            = each.key
  disable_on_destroy = false
}


resource "google_bigquery_dataset" "curated" {
  dataset_id                 = var.dataset_id
  location                   = "US"# CRITICAL: Must match the public data location
  delete_contents_on_destroy = true
}

resource "google_artifact_registry_repository" "repo" {
  location      = var.region
  repository_id = var.artifact_repo
  format        = "DOCKER"
}

resource "google_service_account" "workload_sa" {
  account_id   = "trend-platform-sa"
  display_name = "Trend Platform Workload SA"
}

resource "google_project_iam_member" "bq_editor" {
  project = var.project_id
  role    = "roles/bigquery.dataEditor"
  # below line allow google_service_account.workload_sa to operate bigquery
  member  = "serviceAccount:${google_service_account.workload_sa.email}"
}

resource "google_project_iam_member" "bq_job_user" {
  project = var.project_id
  role    = "roles/bigquery.jobUser"
  member  = "serviceAccount:${google_service_account.workload_sa.email}"
}

# Create the Kubernetes Service Account (KSA)
resource "kubernetes_service_account_v1" "workload_sa" {
  metadata {
    name      = "trend-sa"
    namespace = "trend-platform"
    annotations = {
      # This tells GKE which Google Service Account to use
      "iam.gke.io/gcp-service-account" = google_service_account.workload_sa.email
    }
  }
}

# Allow the KSA to use the GSA(google service account) via Workload Identity
resource "google_service_account_iam_member" "workload_identity_binding" {
  service_account_id = google_service_account.workload_sa.name
  role               = "roles/iam.workloadIdentityUser"
  # Format: serviceAccount:PROJECT_ID.svc.id.goog[NAMESPACE/KSA_NAME]
  member             = "serviceAccount:${var.project_id}.svc.id.goog[trend-platform/trend-sa]"
}

resource "google_container_cluster" "autopilot" {
  name     = var.cluster_name
  location = var.region

  enable_autopilot = true
}
