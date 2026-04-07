What we are building

A small cloud-native analytics platform with two workloads:

API service on GKE
serves curated trend summaries
reads from your own BigQuery dataset
Worker jobs on GKE
process partitions of bigquery-public-data.google_trends.top_rising_terms
write curated results into your BigQuery dataset

Architecture
BigQuery public dataset
  `bigquery-public-data.google_trends.top_rising_terms`
                │
                ▼
        GKE worker Jobs
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

Tech used:

Terraform creates:

GKE Autopilot cluster,BigQuery dataset,Artifact Registry repo,service account + IAM
API container deployed to GKE
Worker container deployed as a Kubernetes Job
Worker reads public data and writes one curated table
API reads curated table and serves 2–3 endpoints


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
