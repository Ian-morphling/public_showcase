{{ config(materialized='table') }}

SELECT 
    FORMAT_DATE('%Y%m%d', d) AS pk_date_sid,
    d AS full_date,
    EXTRACT(YEAR FROM d) AS year,
    EXTRACT(MONTH FROM d) AS month,
    FORMAT_DATE('%B', d) AS month_name,
    EXTRACT(DAY FROM d) AS day_of_month,
    FORMAT_DATE('%A', d) AS day_name,
    CASE WHEN FORMAT_DATE('%A', d) IN ('Sunday', 'Saturday') THEN 0 ELSE 1 END AS is_weekday,
    FORMAT_TIMESTAMP('%Y-%m-%d %I:%M:%S %p', CURRENT_TIMESTAMP()) AS load_date
FROM UNNEST(GENERATE_DATE_ARRAY('2014-01-01', '2050-01-01', INTERVAL 1 DAY)) AS d 
