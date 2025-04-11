from pyspark import SparkContext
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import StructType, StructField, IntegerType, ArrayType
import sys

def find_shortest_path(spark, edges_df, start_id, finish_id, max_iterations=200):
    """
    Поиск кратчайшего пути с использованием BFS на Spark DataFrame
    
    Args:
        spark: SparkSession
        edges_df: DataFrame с рёбрами графа [user_id, follower_id]
        start_id: Начальный узел
        finish_id: Целевой узел
        max_iterations: Максимальная глубина поиска
        
    Returns:
        DataFrame с колонками [node, path, distance]
    """
    # Проверка граничного случая: старт и финиш совпадают
    if start_id == finish_id:
        result_schema = StructType([
            StructField("node", IntegerType()),
            StructField("path", ArrayType(IntegerType())),
            StructField("distance", IntegerType())
        ])
        return spark.createDataFrame([(start_id, [start_id], 0)], schema=result_schema)
    
    paths_schema = StructType([
        StructField("node", IntegerType()),
        StructField("path", ArrayType(IntegerType())),
        StructField("distance", IntegerType())
    ])
    

    paths_df = spark.createDataFrame(
        [(start_id, [start_id], 0)],
        schema=paths_schema
    ).cache()
    

    for iteration in range(max_iterations):

        current_frontier = paths_df.filter(F.col("distance") == iteration)
        

        if current_frontier.rdd.isEmpty():
            break

        new_paths = (
            current_frontier
            .withColumnRenamed("node", "follower_id")
            .join(edges_df, "follower_id", "inner")
            .select(
                F.col("user_id").alias("node"),
                F.array_append("path", "user_id").alias("path"),
                (F.col("distance") + 1).alias("distance")
            )
        )
        

        found_paths = new_paths.filter(F.col("node") == finish_id)
        if not found_paths.rdd.isEmpty():
            return found_paths
        

        existing_nodes = paths_df.select("node")
        new_paths = new_paths.join(existing_nodes, "node", "left_anti")
        

        if new_paths.rdd.isEmpty():
            break
        

        paths_df = paths_df.unionByName(new_paths).cache()
    
 
    return spark.createDataFrame([], paths_schema)

def main():

    conf = SparkConf().setAppName("Shortest Path")
    sc = SparkContext(conf=conf)
    spark = SparkSession(sc)
    

    start_node = int(sys.argv[1])
    finish_node = int(sys.argv[2])
    input_path = sys.argv[3]
    output_path = sys.argv[4]
    

    edge_schema = StructType([
        StructField("user_id", IntegerType()),
        StructField("follower_id", IntegerType())
    ])
    

    edges_df = spark.read.csv(
        input_path,
        schema=edge_schema,
        sep="\t"
    ).cache()
    

    result_df = find_shortest_path(spark, edges_df, start_node, finish_node)
    
    # Форматирование результата
    if not result_df.rdd.isEmpty():
        formatted_result = result_df.select(
            F.concat_ws("->", "path").alias("path")
        )
    else:
        formatted_result = spark.createDataFrame([], "path string")
    
    # Сохранение результатов
    formatted_result.coalesce(1).write.mode("overwrite").text(output_path)
    
    # Остановка Spark
    spark.stop()

if __name__ == "__main__":
    main()