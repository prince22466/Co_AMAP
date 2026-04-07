from google.cloud import bigquery


def run_partitioned_load() -> None:
    client = bigquery.Client()
    query = """
    -- TODO: parameterize date windows and destination table
    SELECT *
    FROM `bigquery-public-data.google_trends.top_rising_terms`
    LIMIT 100
    """

    # `.result()` blocks until the query finishes.
    # In this scaffold, results are only executed/read by the job and are NOT yet
    # written to a curated BigQuery destination table (that write step is a TODO).
    _ = client.query(query).result()


if __name__ == "__main__":
    run_partitioned_load()
