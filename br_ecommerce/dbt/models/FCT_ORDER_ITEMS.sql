{{ config(materialized='table') }}

SELECT  
    I.order_id as fk_order_sid,
    I.order_id as pk_order_id,
    CAST(order_item_id AS STRING) AS pk_order_item_id,
    FORMAT_DATE('%Y%m%d', I.shipping_limit_date) AS fk_shipping_limit_date_sid,
    I.product_id,
    I.seller_id,
    I.shipping_limit_date,
    I.price,
    I.freight_value,
    P.product_category_name,
    C.product_category_name_english,
    P.product_name_lenght as product_name_length,
    P.product_description_lenght as product_description_length,
    P.product_photos_qty,
    P.product_weight_g,
    P.product_length_cm,
    P.product_height_cm,
    P.product_width_cm,
    LPAD(CAST(S.seller_zip_code_prefix AS STRING), 5, '0') AS seller_zip_code_prefix,
    S.seller_city,
    S.seller_state,
    FORMAT_TIMESTAMP('%Y-%m-%d %I:%M:%S %p', CURRENT_TIMESTAMP()) AS load_date
FROM {{ source('olist_brazilian_ecommerce', 'public_order_items') }} I
LEFT JOIN {{ source('olist_brazilian_ecommerce', 'public_products') }} P ON I.product_id = P.product_id
LEFT JOIN {{ source('olist_brazilian_ecommerce', 'public_product_category_name_translation') }} C ON P.product_category_name = C.product_category_name
LEFT JOIN {{ source('olist_brazilian_ecommerce', 'public_sellers') }} S ON I.seller_id = S.seller_id 
