from dagster import Definitions, ScheduleDefinition
from dagster_meltano import meltano_resource
from dagster import define_asset_job

from meltano_orchestration.assets import (
    supabase_postgres_to_bigquery,
    run_dbt_models,
    run_dbt_tests,
    run_feature_engineering_script,
    run_great_expectations_tests,
)

asset_job = define_asset_job(name="elt_pipeline")

# Define schedule for the asset job
daily_schedule = ScheduleDefinition(
    job=asset_job,
    cron_schedule="0 0 * * *",
)

defs = Definitions(
    assets=[
        supabase_postgres_to_bigquery,
        run_dbt_models,
        run_dbt_tests,
        run_feature_engineering_script,
        run_great_expectations_tests,
    ],
    resources={
        "meltano": meltano_resource.configured({
            "project_dir": "/home/chuhao/dsai_sctp/br_e_commerce/meltano_br"
        })
    },
    schedules=[daily_schedule]
)