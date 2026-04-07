-- Top terms by latest date
SELECT country, term, rank, score
FROM `{{project_id}}.{{dataset_id}}.curated_trends`
WHERE event_date = (SELECT MAX(event_date) FROM `{{project_id}}.{{dataset_id}}.curated_trends`)
ORDER BY rank
LIMIT 100;
