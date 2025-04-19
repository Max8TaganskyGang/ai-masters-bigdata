import argparse
from pyspark.sql import SparkSession
import sys

spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel('WARN')

model_path = sys.argv[1]
test_path = sys.argv[2]
output_path = sys.argv[3]

#from pyspark.ml import PipelineModel
#model = PipelineModel.load(model_path)
from mofel import pipeline

test_df = spark.read.json(test_path)
predictions = pipeline.transform(test_df)

predictions.select("id", "prediction").write.mode("overwrite").csv(output_path)