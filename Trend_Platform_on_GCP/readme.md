# Trend Platform on GCP

## Overview
This project builds a **cloud-native analytics platform** on Google Cloud with two core workloads running on GKE:

1. **Worker jobs (GKE Job/CronJob)**
   - Process partitions of `bigquery-public-data.google_trends.top_rising_terms`.
   - Write curated trend outputs into your BigQuery dataset.
2. **API service (GKE Deployment)**
   - Serves curated trend summaries as a JSON API.
   - Reads from curated tables in your own BigQuery dataset.


---

## Architecture

```text
BigQuery public dataset
  bigquery-public-data.google_trends.top_rising_terms
                │
                ▼
        GKE Worker Jobs
     (parallel by date range)
                │
                ▼
   BigQuery curated tables/views
                │
                ▼
         GKE API Deployment
                │
                ▼
            JSON API
```

---

## Tech Stack and Infrastructure

### Terraform provisions
- GKE Autopilot cluster
- BigQuery dataset
- Artifact Registry repository
- Service account and IAM bindings

### Application deployment
- Worker container deployed as a Kubernetes Job/CronJob
- Worker reads public trend data and writes curated tables
- API container deployed to GKE
- API reads curated data and exposes endpoints

---

## Repository Structure

```text
trend-platform/
  README.md

  infra/
    main.tf
    variables.tf
    outputs.tf
    terraform.tfvars.example
    providers.tf

  app/
    api/
      main.py
      requirements.txt
      Dockerfile
    worker/
      worker.py
      requirements.txt
      Dockerfile

  k8s/
    namespace.yaml
    api-deployment.yaml
    api-service.yaml
    worker-job.yaml
    worker-cronjob.yaml

  sql/
    curated_schema.sql
    sample_queries.sql

  scripts/
    deploy.py
    build_and_push.py
    run_job.py
    smoke_test.py
```
