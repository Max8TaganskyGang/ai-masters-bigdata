import argparse
from pyspark.sql import SparkSession
import sys


spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel('WARN')

train_path = sys.argv[1]
model_path = sys.argv[2]


from model import pip
train_df = spark.read.json(train_path).fillna( {"reviewText": "missingreview"})
pipeline_model = pip.fit(train_df)

pipeline_model.write().overwrite().save(model_path)