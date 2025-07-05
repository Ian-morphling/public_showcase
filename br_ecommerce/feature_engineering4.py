import duckdb
import pandas as pd
from pandas_gbq import to_gbq
from pandas_gbq import read_gbq

def run_feature_engineering_script2():
    try:
        # query bigquery
        project_id = "projectm2-aiess"
        query = "SELECT * FROM olist_brazilian_ecommerce_target.DIM_CUSTOMERS"
        df_customers = read_gbq(query, project_id=project_id)
        query2 = "SELECT * FROM olist_brazilian_ecommerce_target.FCT_PAYMENTS"
        df_payments = read_gbq(query2, project_id=project_id)
        query3 = "SELECT * FROM olist_brazilian_ecommerce_target.FCT_REVIEWS"
        df_reviews = read_gbq(query3, project_id=project_id)
        query4 = "SELECT * FROM olist_brazilian_ecommerce_target.DIM_GEOLOCATION"
        df_geolocation = read_gbq(query4, project_id=project_id)
        query5 = "SELECT * FROM olist_brazilian_ecommerce_target.FCT_ORDER_ITEMS"
        df_items =  read_gbq(query5, project_id=project_id)
        query6 = "SELECT * FROM olist_brazilian_ecommerce_target.DIM_DATE"
        df_date = read_gbq(query6, project_id=project_id)
        query7 = "SELECT * FROM olist_brazilian_ecommerce_target.DIM_ORDERS"
        df_orders = read_gbq(query7, project_id=project_id)

        DS_orders_delivery = df_orders.copy()
        # Filter for only delivered orders
        DS_orders_delivery = DS_orders_delivery[DS_orders_delivery['order_status'] == 'delivered']
        # Compute delivery times
        DS_orders_delivery['actual_delivery_time'] = DS_orders_delivery['order_delivered_customer_date'] - DS_orders_delivery['order_purchase_timestamp']
        DS_orders_delivery['estimated_delivery_time'] = DS_orders_delivery['order_estimated_delivery_date'] - DS_orders_delivery['order_purchase_timestamp']
        # Convert to minutes
        DS_orders_delivery['actual_delivery_time_minutes'] = DS_orders_delivery['actual_delivery_time'].dt.total_seconds() / 60
        DS_orders_delivery['estimated_delivery_time_minutes'] = DS_orders_delivery['estimated_delivery_time'].dt.total_seconds() / 60

        to_gbq(
            DS_orders_delivery,
            destination_table='olist_brazilian_ecommerce_DS.DS_orders_delivery',
            project_id='projectm2-aiess',
            if_exists='replace',
        )

        #get buyer city + orders in df
        orders_customers = pd.merge(
            df_orders,
            df_customers,
            how='left',
            left_on='fk_customer_sid',
            right_on='pk_customer_sid'
        )

        #group by order instead to avoid dup on merge
        #using mode since there are orders with multiple product_category_english
        item_agg = df_items.groupby('fk_order_sid').agg({
            'price': 'sum',
            'product_category_name_english': lambda x: x.mode().iloc[0] if not x.mode().empty else None
        }).reset_index()
        orders_full = pd.merge(df_orders, item_agg, left_on='pk_order_sid', right_on='fk_order_sid', how='left')
        orders_full['profit'] = orders_full['total_payment'] - orders_full['price']

        to_gbq(
            orders_full,
            destination_table='olist_brazilian_ecommerce_DS.DS_orders_full_profits',
            project_id='projectm2-aiess',
            if_exists='replace',
        )

        DS_land_geolocation = df_geolocation.copy()

        DS_land_geolocation = DS_land_geolocation[
            (DS_land_geolocation['geolocation_lat'].between(-34, 5)) &
            (DS_land_geolocation['geolocation_lng'].between(-74, -34))
        ]

        to_gbq(
            DS_land_geolocation,
            destination_table='olist_brazilian_ecommerce_DS.DS_land_geolocation',
            project_id='projectm2-aiess',
            if_exists='replace',
        )

        return "Feature engineering completed successfully"
    except Exception as e:
        print(f"feature_engineering.py failed with error: {e}")
        #avoid silent failure
        raise

if __name__ == "__main__":
    run_feature_engineering_script2()