-- Import prompts data (only do this once)
-- .mode csv
-- .import <PROMPTS_CSV_PATH> prompts

CREATE TEMP TABLE training_data AS
SELECT *
FROM responses AS R
WHERE R.filename NOT IN (SELECT P.filename FROM prompts AS P);

CREATE TEMP TABLE training_data_toxicity_percentile AS
SELECT T.filename          AS filename,
       NTILE(100) OVER win AS percentile
FROM training_data AS T
    WINDOW win AS (ORDER BY T.toxicity);

-- Output CSV files
.mode csv
.headers on

.output finetune_lte2.csv
SELECT T.filename AS filename
FROM training_data_toxicity_percentile AS T
WHERE T.percentile <= 2;

.output finetune_mid20_subsample.csv
SELECT T.filename AS filename
FROM training_data_toxicity_percentile AS T
WHERE T.percentile >= 40 AND T.percentile < 60
ORDER BY random()
LIMIT 150000;

.output finetune_gte99.csv
SELECT T.filename AS filename
FROM training_data_toxicity_percentile AS T
WHERE T.percentile >= 99;
