{{ config(materialized='table') }}

SELECT
    order_id as fk_order_sid,
    order_id as pk_order_id,
    review_id as pk_review_id,
    FORMAT_DATE('%Y%m%d', review_creation_date) AS fk_review_creation_date_sid,
    review_score,
    review_comment_title,
    review_comment_message,
    review_creation_date,
    review_answer_timestamp,
    FORMAT_TIMESTAMP('%Y-%m-%d %I:%M:%S %p', CURRENT_TIMESTAMP()) AS load_date
FROM {{ source('olist_brazilian_ecommerce', 'public_order_reviews') }}
