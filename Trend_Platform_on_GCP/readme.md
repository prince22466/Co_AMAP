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
