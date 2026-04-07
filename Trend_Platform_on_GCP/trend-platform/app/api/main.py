# FastAPI is used here as a lightweight, typed framework for JSON APIs with auto docs.
from fastapi import FastAPI

app = FastAPI(title="Trend Platform API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/trends/top")
def top_trends(limit: int = 10):
    # TODO: read curated data from BigQuery
    return {"limit": limit, "items": []}


@app.get("/trends/by-country/{country}")
def by_country(country: str, limit: int = 10):
    # TODO: read curated data from BigQuery
    return {"country": country, "limit": limit, "items": []}
