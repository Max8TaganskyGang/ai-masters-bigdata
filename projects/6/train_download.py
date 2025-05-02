#!/opt/conda/envs/dsenv/bin/python
import os, sys

from pyspark.sql import SparkSession
from pyspark.sql.types import *

from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import col

path_in, path_out = sys.argv[1], sys.argv[2]

spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel('WARN')

schema = StructType([
    StructField("label", DoubleType()),
    StructField("features", ArrayType(DoubleType(), False))
])

df = spark.read.parquet(path_in)

df = df.select(['label'] + [col("features")[i].alias("f_" + str(i)) for i in range(100)])

df.toPandas().to_csv(path_out, sep=',', index=False)