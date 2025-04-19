import argparse
from pyspark.sql import SparkSession
import sys

spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel('WARN')

from pyspark.ml import PipelineModel
model = PipelineModel.load(sys.argv[1])

test_df = spark.read.json(sys.argv[2])
predictions = pipeline.transform(test_df)

predictions.select("id", "prediction").write.mode("overwrite").csv(sys.argv[3])