from dagster import Definitions, define_asset_job, ScheduleDefinition
from dagster_meltano import meltano_resource

from meltano_orchestration.assets import (
    supabase_postgres_to_bigquery,
    run_dbt_models,
    run_dbt_tests,
    run_feature_engineering_script,
    run_great_expectations_tests,
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
    }
)