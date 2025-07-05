{{ config(materialized='table') }}

SELECT
    customer_id as pk_customer_sid,
    customer_unique_id,
    LPAD(CAST(customer_zip_code_prefix AS STRING), 5, '0') AS customer_zip_code_prefix,
    customer_city,
    customer_state,
    FORMAT_TIMESTAMP('%Y-%m-%d %I:%M:%S %p', CURRENT_TIMESTAMP()) AS load_date
FROM {{ source('olist_brazilian_ecommerce', 'public_customers') }}
