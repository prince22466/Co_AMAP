# FastAPI is used here as a lightweight, typed framework for JSON APIs with auto docs.
from fastapi import FastAPI
from google.cloud import bigquery

app = FastAPI(title="Trend Platform API")
source_db = 'tests101-483015.trend_curated.top_rising_terms'


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/trends/top10")
def top10_trends(limit: int = 10):
    with bigquery.Client(project="tests101-483015") as client:
        query = f"""
                SELECT 
                    trend_item, 
                    SUM(search_volume) AS total_volume
                FROM 
                    `{source_db}`
                WHERE 
                    date = (SELECT MAX(date) FROM `{source_db}`)
                GROUP BY 
                    trend_item
                ORDER BY 
                    total_volume DESC
                LIMIT {limit}
        """
        
        query_job = client.query(query)
        items = [dict(row) for row in query_job]
        
    return {"limit": limit, "items": items}


@app.get("/trends_top10/by-country/{country}")
def by_country(country: str, limit: int = 10):
    with bigquery.Client(project="tests101-483015") as client:
        query = f"""
                SELECT 
                    trend_item
                FROM 
                    `{source_db}`
                WHERE 
                    week_date = (SELECT MAX(datweek_datee) FROM `{source_db}`)
                    And country = @country
                ORDER BY 
                    rank DESC
                LIMIT {limit}
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("country", "STRING", country)
            ]
        ) ## tells BigQuery, "Whenever you see @country, replace it with this string value I'm giving you."
          ## @country is to prevent malicious injection by user

        query_job = client.query(query, job_config=job_config)
        items = [dict(row) for row in query_job]
    
    return {"country": country, "limit": limit, "items": items}
