-- BigQuery table in your curated dataset: {{project_id}}.{{dataset_id}}.curated_trends
CREATE TABLE IF NOT EXISTS `{{project_id}}.{{dataset_id}}.curated_trends` (
  week_date DATE,
  country STRING,
  trend_item STRING,
  rank INT64,
)
PARTITION BY week_date
CLUSTER BY country;
