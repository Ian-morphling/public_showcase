{{ config(materialized='table') }}

SELECT
    order_id as fk_order_sid,
    order_id as pk_order_id,
    payment_sequential as pk_payment_sequential,
    payment_type,
    payment_installments,
    payment_value,
    FORMAT_TIMESTAMP('%Y-%m-%d %I:%M:%S %p', CURRENT_TIMESTAMP()) AS load_date
FROM {{ source('olist_brazilian_ecommerce', 'public_order_payments') }}
