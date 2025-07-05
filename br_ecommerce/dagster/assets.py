import subprocess
from dagster import AssetExecutionContext, asset
import os

# Constants
MELTANO_PROJECT_ROOT = "/home/chuhao/dsai_sctp/br_e_commerce/meltano_br"
DBT_PROJECT_ROOT = "/home/chuhao/dsai_sctp/br_e_commerce/DBT_JOB_v1.0/olist_brazillian_ecommerce"
GX_SCRIPT_PATH = "/home/chuhao/dsai_sctp/br_e_commerce/meltano-orchestration"
GX_SCRIPT_FILE = "GX_tests_comb2.py"
FEATURE_ENGINEERING_SCRIPT_PATH = "/home/chuhao/dsai_sctp/br_e_commerce/meltano-orchestration"
FEATURE_ENGINEERING_SCRIPT_FILE = "feature_engineering4.py"

@asset
def supabase_postgres_to_bigquery(context: AssetExecutionContext):
    context.log.info(" Running Meltano pipeline: tap-postgres -> target-bigquery")

    result = subprocess.run(
        ["meltano", "run", "tap-postgres", "target-bigquery", "--force"],
        cwd=MELTANO_PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    context.log.info(f" Meltano return code: {result.returncode}")
    context.log.info(" STDOUT:\n" + result.stdout)
    if result.stderr:
        context.log.error(" STDERR:\n" + result.stderr)

    if result.returncode != 0:
        raise Exception(
            f"Meltano run failed with return code {result.returncode}.\n"
            f"STDERR:\n{result.stderr}\n"
            f"STDOUT:\n{result.stdout}"
        )  

@asset(deps=[supabase_postgres_to_bigquery])
def run_dbt_models(context: AssetExecutionContext):
    context.log.info(" Running `dbt run`")

    result = subprocess.run(
        ["dbt", "run"],
        cwd=DBT_PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    context.log.info(f" dbt run return code: {result.returncode}")
    context.log.info(" STDOUT:\n" + result.stdout)
    if result.stderr:
        context.log.error(" STDERR:\n" + result.stderr)

    if result.returncode != 0:
        raise Exception(
            f"dbt run failed with return code {result.returncode}.\n"
            f"STDERR:\n{result.stderr}\n"
            f"STDOUT:\n{result.stdout}"
        )   

@asset(deps=[run_dbt_models])
def run_dbt_tests(context: AssetExecutionContext):
    context.log.info(" Running `dbt test`")

    result = subprocess.run(
        ["dbt", "test"],
        cwd=DBT_PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    context.log.info(f" dbt test return code: {result.returncode}")
    context.log.info(" STDOUT:\n" + result.stdout)
    if result.stderr:
        context.log.error(" STDERR:\n" + result.stderr)

    if result.returncode != 0:
        raise Exception(
            f"dbt test failed with return code {result.returncode}.\n"
            f"STDERR:\n{result.stderr}\n"
            f"STDOUT:\n{result.stdout}"
        )      

@asset(deps=[run_dbt_tests])
def run_feature_engineering_script(context: AssetExecutionContext):
    context.log.info(" Running feature_engineering")

    result = subprocess.run(
        ["python", FEATURE_ENGINEERING_SCRIPT_FILE],
        cwd=FEATURE_ENGINEERING_SCRIPT_PATH,
        capture_output=True,
        text=True,
    )

    context.log.info(" STDOUT:\n" + result.stdout)
    if result.stderr:
        context.log.error(" STDERR:\n" + result.stderr)

    if result.returncode != 0:
        raise Exception(
            f"feature_engineering.py failed with return code {result.returncode}.\n"
            f"STDERR:\n{result.stderr}\n"
            f"STDOUT:\n{result.stdout}"
        )  
    
@asset(deps=[run_feature_engineering_script])
def run_great_expectations_tests(context: AssetExecutionContext):
    context.log.info(" Running GX_tests")

    result = subprocess.run(
        ["python", GX_SCRIPT_FILE],
        cwd=GX_SCRIPT_PATH,
        capture_output=True,
        text=True,
    )

    context.log.info(" STDOUT:\n" + result.stdout)
    if result.stderr:
        context.log.error(" STDERR:\n" + result.stderr)

    if result.returncode != 0:
        raise Exception(
            f"GX_tests.py failed with return code {result.returncode}.\n"
            f"STDERR:\n{result.stderr}\n"
            f"STDOUT:\n{result.stdout}"
        )