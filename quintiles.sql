-- Responses
SELECT 
    responses.filename AS filename,
    NTILE(5) OVER win AS quintile
FROM 
    responses
WINDOW
    win AS (ORDER BY responses.toxicity);

-- Create table for responses quintiles
CREATE TABLE responses_quintiles AS
    SELECT
        -- Primary key
        responses.filename AS filename,
        -- Quintile value
        NTILE(5) OVER win AS quintile
    FROM 
        responses
    WINDOW
        win AS (ORDER BY responses.toxicity);


-- Span Scores
SELECT
    span_scores.*, 
    NTILE(5) OVER win AS quintile
FROM 
    span_scores
WINDOW 
    win AS (ORDER BY span_scores.toxicity);


-- Create table for span score quintiles
CREATE TABLE span_scores_quintiles AS
    SELECT
        -- Primary key (composite)
        span_scores.filename    AS filename,
        span_scores.begin       AS begin,
        span_scores.end         AS end,
        -- Quintile value
        NTILE(5) OVER win       AS quintile
    FROM 
        span_scores
    WINDOW 
        win AS (ORDER BY span_scores.toxicity);