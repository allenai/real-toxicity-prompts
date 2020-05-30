-- Subsample docs
CREATE TEMP TABLE docs_out AS
SELECT *
FROM responses
ORDER BY random()
LIMIT 100000;

-- Find associated spans
CREATE TEMP TABLE spans_out AS
SELECT S.*
FROM span_scores AS S
         INNER JOIN
     docs_out AS D
     ON S.filename = D.filename;

-- Output CSV files
.mode csv
.headers on

.output docs.csv
SELECT *
from docs_out;

.output spans.csv
SELECT *
FROM spans_out;
