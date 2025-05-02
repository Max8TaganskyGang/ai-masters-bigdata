#!/opt/conda/envs/dsenv/bin/python

import sys, os
from joblib import load
import pandas as pd

sys.path.append('.')

from pyspark.sql import SparkSession
from pyspark.sql.types import *

from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import col
from pyspark.sql.functions import pandas_udf

test_in, pred_out, model_in = sys.argv[1], sys.argv[2], sys.argv[3]

spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel('WARN')

schema = StructType([
    StructField("id", LongType()),
    StructField("features", ArrayType(DoubleType(), False))
])

df = spark.read.parquet(test_in)
df = df.select(['id'] + [col("features")[i].alias("f_" + str(i)) for i in range(100)])

model = load(model_in)
sparkModel = spark.sparkContext.broadcast(model)

@pandas_udf('double')
def predict_pandas_udf(*cols):
    X = pd.concat(cols, axis=1)
    X.columns = [f"f_{i}" for i in range(len(X.columns))]  # Приводим к ожидаемым именам
    return pd.Series(sparkModel.value.predict(X))

list_of_columns = df.columns[1:]
df = df.withColumn('prediction', predict_pandas_udf(*list_of_columns))

df.select("id", "prediction").write.mode("overwrite").csv(pred_out)