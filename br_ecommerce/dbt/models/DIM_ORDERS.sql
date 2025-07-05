{{ config(materialized='table') }}

WITH payment AS (
  SELECT order_id, SUM(payment_value) AS total_payment
  FROM {{ source('olist_brazilian_ecommerce', 'public_order_payments') }}
  GROUP BY order_id
),
order_items AS (
  SELECT order_id, 
  SUM(price) order_amt, 
  SUM(freight_value) as freight_amt, 
  SUM(price) + sum(freight_value) as total_order_amt_wf_freight
  FROM {{ source('olist_brazilian_ecommerce', 'public_order_items') }}
  GROUP BY order_id
)
SELECT 
  o.order_id AS pk_order_sid, 
  o.customer_id AS fk_customer_sid, 
  FORMAT_DATE('%Y%m%d', CAST(o.order_purchase_timestamp AS DATE)) AS fk_order_purchased_date_sid,
  CASE 
    WHEN order_delivered_customer_date IS NOT NULL AND o.order_approved_at != '' 
    THEN FORMAT_DATE('%Y%m%d', DATE(o.order_approved_at))
    ELSE NULL 
  END AS fk_order_approved_at_date_sid, 
  CASE 
    WHEN order_delivered_customer_date IS NOT NULL AND o.order_delivered_carrier_date != '' 
    THEN FORMAT_DATE('%Y%m%d', DATE(o.order_delivered_carrier_date))
    ELSE NULL 
  END AS fk_order_delivered_carrier_date_sid, 
  CASE 
    WHEN order_delivered_customer_date IS NOT NULL AND o.order_delivered_customer_date != '' 
    THEN FORMAT_DATE('%Y%m%d', DATE(o.order_delivered_customer_date))
    ELSE NULL 
  END AS fk_order_delivered_customer_date_sid, 
  FORMAT_DATE('%Y%m%d', CAST(o.order_estimated_delivery_date AS DATE)) AS fk_order_estimated_delivery_date_sid, 
  o.order_status AS order_status,
  ROUND(p.total_payment, 2) AS total_payment, 
  ROUND(i.order_amt, 2) AS order_amt, 
  ROUND(i.freight_amt, 2) AS freight_amt, 
  ROUND(i.total_order_amt_wf_freight, 2) AS total_order_amt_wf_freight, 
  ROUND(i.total_order_amt_wf_freight - p.total_payment, 2) AS balance_amt, 
  CASE 
    WHEN p.total_payment>0 and i.order_amt>0 and i.total_order_amt_wf_freight - p.total_payment = 0 THEN 'completed'
    WHEN p.total_payment>0 and i.order_amt>0 and i.total_order_amt_wf_freight - p.total_payment < 0 THEN 'completed'
    WHEN p.total_payment>0 and i.order_amt>0 and i.total_order_amt_wf_freight - p.total_payment > 0 THEN 'in progress'
    WHEN p.total_payment > 0 and i.order_amt IS NULL THEN 'order items not found'
    WHEN o.order_status in ('delivered','shipped') and (p.total_payment IS NULL OR p.total_payment <= 0) THEN 'payments not found'
    WHEN o.order_status in ('canceled','unavailable') and (p.total_payment IS NULL OR p.total_payment <= 0) THEN 'canceled'
    ELSE NULL
  END AS payment_status,
  SAFE_CAST(o.order_purchase_timestamp AS TIMESTAMP) AS order_purchase_timestamp,
  SAFE.PARSE_TIMESTAMP('%Y-%m-%d %H:%M:%S', o.order_approved_at) AS order_approved_at,
  SAFE.PARSE_TIMESTAMP('%Y-%m-%d %H:%M:%S', o.order_delivered_carrier_date) AS order_delivered_carrier_date,
  SAFE.PARSE_TIMESTAMP('%Y-%m-%d %H:%M:%S', o.order_delivered_customer_date) AS order_delivered_customer_date,
  SAFE_CAST(o.order_estimated_delivery_date AS TIMESTAMP) AS order_estimated_delivery_date,
  FORMAT_TIMESTAMP('%Y-%m-%d %I:%M:%S %p', CURRENT_TIMESTAMP()) AS load_date
FROM {{ source('olist_brazilian_ecommerce', 'public_orders') }} o
LEFT JOIN payment p ON o.order_id = p.order_id
LEFT JOIN order_items i ON o.order_id = i.order_id 
