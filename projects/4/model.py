from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import *
from pyspark.ml import Pipeline

tokenizer = Tokenizer(inputCol="reviewText", outputCol="words")
hasher = HashingTF(numFeatures=50, binary=True, inputCol=tokenizer.getOutputCol(), outputCol="word_vector")

lr = LinearRegression(featuresCol=hasher.getOutputCol(), labelCol="overall", maxIter=15)

pipeline = Pipeline(stages=[
    tokenizer,
    hasher,
    lr
])

