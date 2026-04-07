-- BigQuery table in your curated dataset: {{project_id}}.{{dataset_id}}.curated_trends
CREATE TABLE IF NOT EXISTS `{{project_id}}.{{dataset_id}}.curated_trends` (
  event_date DATE,
  country STRING,
  term STRING,
  rank INT64,
  score FLOAT64,
  loaded_at TIMESTAMP
)
PARTITION BY event_date
CLUSTER BY country;
