import json
from pathlib import Path

from google.cloud import bigquery

SETTINGS_PATH = Path(__file__).resolve().with_name("settings.json")

with SETTINGS_PATH.open("r", encoding="utf-8") as f:
    settings = json.load(f)

project_id = settings["gcp"]["project_id"]
src_dtable = settings["bigquery"]["source_table"]
target_table = settings["bigquery"]["target_table"]


def run_partitioned_load() -> None:
    client = bigquery.Client(project=project_id)

    job_config = bigquery.QueryJobConfig(
        destination=target_table,
        write_disposition="WRITE_TRUNCATE",  # Overwrites the table. Use WRITE_APPEND to add rows instead.
    )

    query = f"""
    SELECT country_name as country,
    week as week_date,
    term as trend_item,
    rank
    FROM `{src_dtable}`
    """

    # Run the query and write the results to the destination table
    query_job = client.query(query, job_config=job_config)
    query_job.result()  # Waits for the job to complete
    client.close()
    print(f"Query results loaded to {target_table}")


if __name__ == "__main__":
    run_partitioned_load()
