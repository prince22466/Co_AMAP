from google.cloud import bigquery

src_dtable = 'bigquery-public-data.google_trends.international_top_terms'
target_table = 'tests101-483015.trend_curated.top_rising_terms'

def run_partitioned_load() -> None:
    client = bigquery.Client(project="tests101-483015")
    
    job_config = bigquery.QueryJobConfig(
        destination=target_table,
        write_disposition="WRITE_TRUNCATE", # Overwrites the table. Use WRITE_APPEND to add rows instead.
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
