from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import *
from pyspark.ml import Pipeline

tok = Tokenizer(inputCol="reviewText", outputCol="words")
pipeline = Pipeline(stages=[
    tok,
    HashingTF(numFeatures=50, binary=True, inputCol=tokenizer.getOutputCol(), outputCol="word_vector"),
    LinearRegression(featuresCol=hasher.getOutputCol(), labelCol="overall", maxIter=15)
])

