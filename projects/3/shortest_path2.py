from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
import sys

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, array, array_append, lit, concat_ws
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, IntegerType

def find_shortest_path(spark, edges_df, start_id, finish_id, max_iterations=20):
    """
    BFS
    
    Args:
        spark: SparkSession
        edges_df: Massive DataFrame with [follower_id, user_id] edges
        start_id: Starting node ID
        finish_id: Target node ID
        max_iterations: Maximum BFS depth to search
        
    Returns:
        DataFrame with shortest paths (empty if no path found)
    """
    edges_df = edges_df.cache()

    paths_schema = StructType([
        StructField("node", IntegerType()),
        StructField("path", ArrayType(IntegerType())),
        StructField("distance", IntegerType())
    ])
    
    paths_df = spark.createDataFrame(
        [(start_id, [start_id], 0)],
        schema=paths_schema
    )
    
    for iteration in range(max_iterations):
        frontier = paths_df.filter(col("distance") == iteration)
        
        frontier_renamed = frontier.selectExpr("node as follower_id", "path", "distance")
        new_paths = edges_df.join(
            frontier_renamed,
            on="follower_id",
            how="inner"
        )
        
        new_paths = new_paths.select(
            col("user_id").alias("node"),
            array_append(col("path"), col("user_id")).alias("path"),
            (col("distance") + 1).alias("distance")
        )
        
        
        finish_paths = new_paths.filter(col("node") == finish_id)
        if not finish_paths.rdd.isEmpty():
            return finish_paths
        
        existing_nodes = paths_df.select("node")
        new_paths = new_paths.join(existing_nodes, "node", "left_anti")
        
        if new_paths.rdd.isEmpty():
            break
            
        paths_df = paths_df.unionByName(new_paths)
    
    return spark.createDataFrame([], paths_schema)

conf = SparkConf()
sc = SparkContext(appName="Pagerank", conf=conf)
spark = SparkSession(sc)

start_node = int(sys.argv[1])
finish_node = int(sys.argv[2])
dataset_path = sys.argv[3]
output_file_path = sys.argv[4]

schema = StructType([
    StructField("user_id", IntegerType()),
    StructField("follower_id", IntegerType())
])

edges = spark.read.csv(dataset_path, schema = schema, sep = "\t")

result = find_shortest_path(spark, edges, start_node, finish_node, max_iterations=200)

valid_paths = result.select(concat_ws(",", col("path")).alias("full_path"))

valid_paths.coalesce(1).write.mode("overwrite").text(output_file_path)