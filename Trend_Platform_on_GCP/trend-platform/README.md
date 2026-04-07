# Trend Platform

Cloud-native analytics platform scaffold for:
- **GKE API service** serving curated trend summaries.
- **GKE worker jobs** reading Google Trends public BigQuery data and writing curated results.

## Structure

- `infra/`: Terraform for GKE Autopilot, BigQuery dataset, Artifact Registry, IAM.
- `app/api/`: FastAPI service container.
- `app/worker/`: BigQuery processing worker container.
- `k8s/`: Kubernetes manifests (Deployment, Service, Job, CronJob).
- `sql/`: Curated schema and sample queries.
- `scripts/`: Build, deploy, run, and smoke-test helpers.

## Notes (from review comments)

- `python:3.11-slim` is the official Python image with a reduced Debian userspace; it keeps image size smaller while still supporting pip installs for this scaffold.
- API uses **FastAPI** because it is lightweight, has built-in OpenAPI docs, and is a common fit for JSON microservices on Kubernetes.
- `infra/outputs.tf` is used to print and expose useful Terraform values after `terraform apply` (for example dataset and cluster identifiers) so scripts or operators can consume them.
- `infra/terraform.tfvars.example` is a template file showing which variables the user should fill before running Terraform; copy to `terraform.tfvars` and edit values for your project.
- Curated table DDL in `sql/curated_schema.sql` is intended for **BigQuery**, under `{{project_id}}.{{dataset_id}}.curated_trends`.
