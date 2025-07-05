import numpy as np
import pandas as pd
from pandas_gbq import read_gbq
import great_expectations as gx
from great_expectations import expectations as gxe
import pprint
import os


def run_GX_customers():
    context = gx.get_context()
    # query bigquery
    project_id = "projectm2-aiess"
    query = "SELECT * FROM olist_brazilian_ecommerce_target.DIM_CUSTOMERS"
    df_customers = read_gbq(query, project_id=project_id)


    data_source_name = "olist.dim_customers"
    data_source = context.data_sources.add_pandas(name=data_source_name)

    # create asset
    data_asset_name = "olist.dim_customers_asset"
    data_asset = data_source.add_dataframe_asset(name=data_asset_name)

    # create batch
    batch_definition_name = "batch_customers_dataframe"
    batch_definition = data_asset.add_batch_definition_whole_dataframe(batch_definition_name)

    batch_parameters = {"dataframe": df_customers}

    new_batch = batch_definition.get_batch(batch_parameters=batch_parameters)

    # Create a new suite for all dimension tables schema validation
    suite_name = "schema_dim_customers_expectation"
    suite = gx.ExpectationSuite(name=suite_name)

    schema_dim_customers_expectation = gx.expectations.ExpectColumnToExist(
        column="pk_customer_sid", column_index=0
    )

    context.suites.add_or_update(suite)
    suite.add_expectation(schema_dim_customers_expectation)

    definition_name = "schema_dim_customers_definition"
    validation_definition = gx.ValidationDefinition(
        data=batch_definition, suite=suite, name=definition_name
    )

    validation_results = validation_definition.run(batch_parameters=batch_parameters)

    # Save full results to file
    output_folder = "gx_output"
    os.makedirs(output_folder, exist_ok=True)
    result_path = os.path.join(output_folder, "gx_results_customers.txt")

    with open(result_path, "w") as f:
        f.write(pprint.pformat(validation_results))

    print(f" Full GX test results saved to {result_path}")

def run_GX_dtype_summary():
    context = gx.get_context()
    # List of GBQ tables and their expected columns with types
    gbq_tables_with_columns_and_types = {
        "olist_brazilian_ecommerce_target.FCT_PAYMENTS": {"fk_order_sid": "string"},
        "olist_brazilian_ecommerce_target.FCT_REVIEWS": {"fk_order_sid": "string"},
        "olist_brazilian_ecommerce_target.DIM_GEOLOCATION": {"geolocation_zip_code_prefix": "string"},
        "olist_brazilian_ecommerce_target.FCT_ORDER_ITEMS": {"fk_order_sid": "string"},
        "olist_brazilian_ecommerce_target.DIM_DATE": {"pk_date_sid": "integer"},
    }

    output_folder = "gx_output"
    os.makedirs(output_folder, exist_ok=True)
    summary_file_path = os.path.join(output_folder, "gx_dtype_summary.txt")

    with open(summary_file_path, "w") as f:
        f.write("Great Expectations Full Validation Results\n")
        f.write("=" * 60 + "\n\n")

    # Iterate over the list of tables and process each one   
        for table_name, expected_columns in gbq_tables_with_columns_and_types.items():
            query = f"SELECT * FROM {table_name}"
            df_table = read_gbq(query, project_id="projectm2-aiess")

            # Generate unique names for data source and asset
            data_source_name = f"{table_name}_data_source"
            asset_name = f"{table_name}_asset"

            # Add data source
            data_source = context.data_sources.add_pandas(name=data_source_name)
            # Add DataFrame asset
            data_asset = data_source.add_dataframe_asset(name=asset_name)
            # Add batch definition
            batch_definition = data_asset.add_batch_definition_whole_dataframe(table_name)
            batch_parameters = {"dataframe": df_table}
            batch = batch_definition.get_batch(batch_parameters=batch_parameters)

            #Create Expectation Suite
            suite_name = f"{table_name}_suite"
            suite = gx.ExpectationSuite(name=suite_name)
            suite = context.suites.add(suite)
            # Add ExpectColumnValuesToBeOfType expectations for each expected column
            for column, column_type in expected_columns.items():
                expectation = gx.expectations.ExpectColumnValuesToBeOfType(
                    column=column, type_=column_type
                )
                suite.add_expectation(expectation)

            print(f" Running validation for: {table_name}")
            validation_definition = gx.ValidationDefinition(
                data=batch_definition, suite=suite, name=f"{table_name}_validation"
            )
            results = validation_definition.run(batch_parameters=batch_parameters)
            print(f"Validation results for {table_name}:")
            print(results)

            f.write(f"Table: {table_name}\n")
            f.write(pprint.pformat(results))
            print(f" Finished validation for: {table_name}")

    print(f"Combined gx dytppe test results saved to {summary_file_path}")

def run_GX_fct_order_items():
    context = gx.get_context()

    # Query the fact table from GBQ
    project_id = "projectm2-aiess"
    fact_table_name = "projectm2-aiess.olist_brazilian_ecommerce_target.FCT_ORDER_ITEMS"
    query = f"SELECT * FROM {fact_table_name}"
    df_fact_table = read_gbq(query, project_id=project_id)

    # Generate unique names for data source and asset
    data_source_name = f"{fact_table_name}_data_source"
    asset_name = f"{fact_table_name}_asset"

    # Add data source
    data_source = context.data_sources.add_pandas(name=data_source_name)

    # Add DataFrame asset
    data_asset = data_source.add_dataframe_asset(name=asset_name)

    # Add batch definition
    batch_definition_name = fact_table_name
    batch_definition = data_asset.add_batch_definition_whole_dataframe(batch_definition_name)

    # Get the batch and print the first few rows
    batch_parameters = {"dataframe": df_fact_table}
    batch = batch_definition.get_batch(batch_parameters=batch_parameters)

    # Add column expectations
    schema_fct_orders_expectation_1 = gx.expectations.ExpectColumnToExist(
        column="fk_order_sid", column_index=0
    )
    schema_fct_orders_expectation_2 = gx.expectations.ExpectColumnToExist(
        column="pk_order_id", column_index=1
    )
    schema_fct_orders_expectation_3 = gx.expectations.ExpectColumnToExist(
        column="pk_order_item_id", column_index=2
    )
    schema_fct_orders_expectation_4 = gx.expectations.ExpectColumnToExist(
        column="fk_shipping_limit_date_sid", column_index=3
    )

    # Create a new suite for the fact table schema validation
    suite_name = "schema_fct_orders_expectation"
    suite = gx.ExpectationSuite(name=suite_name)
    suite = context.suites.add(suite)

    suite.add_expectation(schema_fct_orders_expectation_1)
    suite.add_expectation(schema_fct_orders_expectation_2)
    suite.add_expectation(schema_fct_orders_expectation_3)
    suite.add_expectation(schema_fct_orders_expectation_4)

    # Create validation definition
    definition_name = "schema_fct_orders_definition"
    validation_definition = gx.ValidationDefinition(
        data=batch_definition, suite=suite, name=definition_name
    )

    # Run validation
    validation_results = validation_definition.run(batch_parameters=batch_parameters)
    print(f"Validation results for {fact_table_name}:")
    print(validation_results)

    # Save full results to file
    output_folder = "gx_output"
    os.makedirs(output_folder, exist_ok=True)
    result_path = os.path.join(output_folder, "gx_results_FCT_ORDER_ITEMS.txt")

    with open(result_path, "w") as f:
        f.write(pprint.pformat(validation_results))

    print(f" Full GX test results saved to {result_path}")

def run_GX_fct_orders_delivery():
    context = gx.get_context()

    # Query the fact table from GBQ
    project_id = "projectm2-aiess"
    fact_table_name = "olist_brazilian_ecommerce_DS.DS_orders_delivery"
    query = f"SELECT * FROM {fact_table_name}"
    df_fact_table = read_gbq(query, project_id=project_id)

    # Generate unique names for data source and asset
    data_source_name = f"{fact_table_name}_data_source"
    asset_name = f"{fact_table_name}_asset"

    # Add data source
    data_source = context.data_sources.add_pandas(name=data_source_name)

    # Add DataFrame asset
    data_asset = data_source.add_dataframe_asset(name=asset_name)

    # Add batch definition
    batch_definition_name = fact_table_name
    batch_definition = data_asset.add_batch_definition_whole_dataframe(batch_definition_name)

    # Get the batch and print the first few rows
    batch_parameters = {"dataframe": df_fact_table}
    batch = batch_definition.get_batch(batch_parameters=batch_parameters)
    print(f"Batch for {fact_table_name}:")
    print(batch.head(4))

    # Add column expectations
    schema_fct_orders_expectation_1 = gx.expectations.ExpectColumnToExist(
        column="pk_order_sid", column_index=0
    )
    schema_fct_orders_expectation_2 = gx.expectations.ExpectColumnToExist(
        column="fk_customer_sid", column_index=1
    )

    # Create a new suite for the fact table schema validation
    suite_name = "schema_fct_orders_expectation"
    suite = gx.ExpectationSuite(name=suite_name)
    suite = context.suites.add(suite)

    suite.add_expectation(schema_fct_orders_expectation_1)
    suite.add_expectation(schema_fct_orders_expectation_2)

    # Create validation definition
    definition_name = "schema_fct_orders_definition"
    validation_definition = gx.ValidationDefinition(
        data=batch_definition, suite=suite, name=definition_name
    )

    # Run validation
    validation_results = validation_definition.run(batch_parameters=batch_parameters)
    print(f"Validation results for {fact_table_name}:")
    print(validation_results)

    # Save full results to file
    output_folder = "gx_output"
    os.makedirs(output_folder, exist_ok=True)
    result_path = os.path.join(output_folder, "gx_results_DS_fct_orders_delivery.txt")

    with open(result_path, "w") as f:
        f.write(pprint.pformat(validation_results))

    print(f" Full GX test results saved to {result_path}")

def run_GX_geo():
    context = gx.get_context()
    # query bigquery
    project_id = "projectm2-aiess"
    query = "SELECT * FROM olist_brazilian_ecommerce_DS.DS_land_geolocation"
    df_geolocation = read_gbq(query, project_id=project_id)

    data_source_name = "geolocation_df"
    data_source = context.data_sources.add_pandas(name=data_source_name)

    # create asset
    data_asset_name = "geolocation_asset"
    data_asset = data_source.add_dataframe_asset(name=data_asset_name)

    # create batch
    batch_definition_name = "batch_geolocation_dataframe"
    batch_definition = data_asset.add_batch_definition_whole_dataframe(batch_definition_name)

    batch_parameters = {"dataframe": df_geolocation}

    new_batch = batch_definition.get_batch(batch_parameters=batch_parameters)

    suite_name = "br_ecom_expectation"
    suite = gx.ExpectationSuite(name=suite_name)

    preset_lat_expectation = gx.expectations.ExpectColumnValuesToBeBetween(
        column="geolocation_lat", min_value=-35, max_value=5
    )

    preset_long_expectation = gx.expectations.ExpectColumnValuesToBeBetween(
        column="geolocation_lng", min_value=-75, max_value=-35
    )

    context.suites.add_or_update(suite)
    suite.add_expectation(preset_lat_expectation)
    suite.add_expectation(preset_long_expectation)

    definition_name = "br_ecom_validation_definition_V2"
    validation_definition = gx.ValidationDefinition(
        data=batch_definition, suite=suite, name=definition_name
    )

    validation_results = validation_definition.run(batch_parameters=batch_parameters)
    print(validation_results)

    # Save full results to file
    output_folder = "gx_output"
    os.makedirs(output_folder, exist_ok=True)
    result_path = os.path.join(output_folder, "gx_results_geo.txt")

    with open(result_path, "w") as f:
        f.write(pprint.pformat(validation_results))

    print(f" Full GX test results saved to {result_path}")


if __name__ == "__main__":
    run_GX_customers()
    run_GX_dtype_summary()
    run_GX_fct_order_items()
    run_GX_fct_orders_delivery()
    run_GX_geo()
