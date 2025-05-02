#!/opt/conda/envs/dsenv/bin/python

from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.decorators import task
from airflow.sensors.filesystem import FileSensor
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator

pyspark_python = "/opt/conda/envs/dsenv/bin/python"
base_dir = '{{ dag_run.conf["base_dir"] if dag_run else "" }}'

with DAG(
    dag_id="Max8TaganskyGang_dag",
    start_date=datetime(2025, 4, 26),
    schedule=None,
    catchup=False,
    description="hw6",
    tags=["hw6"],
) as dag:
    
    feature_eng_train_task = SparkSubmitOperator(
        task_id="feature_eng_train",
        application=f"{base_dir}/feature_eng.py",
        spark_binary="/usr/bin/spark3-submit",
        num_executors=10,
        executor_cores=1,
        executor_memory="2G",
        application_args=["/datasets/amazon/amazon_extrasmall_train.json", "Max8TaganskyGang_train_out"],
        env_vars={"PYSPARK_PYTHON" : pyspark_python}
    )

    download_train_task = SparkSubmitOperator(
        task_id="download_train",
        application=f"{base_dir}/train_download.py",
        spark_binary="/usr/bin/spark3-submit",
        application_args=["Max8TaganskyGang_train_out", f"{base_dir}/Max8TaganskyGang_train_out_local"],
        env_vars={"PYSPARK_PYTHON" : pyspark_python}
    )

    train_task = BashOperator(
        task_id="train_model",
        bash_command=f"/opt/conda/envs/dsenv/bin/python {base_dir}/train.py {base_dir}/Max8TaganskyGang_train_out_local {base_dir}/6.joblib"
    )

    model_sensor = FileSensor(
        task_id=f'sensor',
        filepath=f"{base_dir}/6.joblib",
        poke_interval=10,
        timeout=2*60
    )

    feature_eng_test_task = SparkSubmitOperator(
        task_id="feature_eng_test",
        application=f"{base_dir}/feature_eng.py",
        spark_binary="/usr/bin/spark3-submit",
        application_args=["/datasets/amazon/amazon_extrasmall_test.json", "Max8TaganskyGang_test_out"],
        env_vars={"PYSPARK_PYTHON" : pyspark_python}
    )

    predict_task = SparkSubmitOperator(
        task_id="predict",
        application=f"{base_dir}/predict.py",
        spark_binary="/usr/bin/spark3-submit",
        application_args=["Max8TaganskyGang_test_out", "Max8TaganskyGang_hw6_prediction", f"{base_dir}/6.joblib"],
        env_vars={"PYSPARK_PYTHON" : pyspark_python}
    )

    feature_eng_train_task >> download_train_task >> train_task >> model_sensor >> feature_eng_test_task >> predict_task
