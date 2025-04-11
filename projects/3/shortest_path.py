from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import StructType, StructField, IntegerType, ArrayType
import sys

def main():
    start_node = int(sys.argv[1])  
    target_node = int(sys.argv[2])  
    input_file = sys.argv[3]        
    output_file = sys.argv[4]      

    spark_config = SparkConf()
    spark_session = SparkSession.builder.config(conf=spark_config).appName("BFS").getOrCreate()

    edge_schema = StructType([
        StructField('user_id', IntegerType(), nullable=False),
        StructField('follower_id', IntegerType(), nullable=False)
    ])
    
    edges_df = spark_session.read.csv(input_file, sep='\t', header=False, schema=edge_schema)
    max_possible_depth = edges_df.select('user_id', 'follower_id').distinct().count()
    
    path_schema = StructType([
        StructField('node_id', IntegerType(), nullable=False),
        StructField('path', ArrayType(IntegerType()), nullable=False)
    ])
    
    frontier = spark_session.createDataFrame([(start_node, [start_node])], path_schema)
    edges_df.cache()

    current_depth = 0

    while current_depth < max_possible_depth:
        discovered_nodes = (
            frontier
            .join(edges_df, frontier['node_id'] == edges_df['follower_id'])
            .filter(~array_contains(col('path'), col('user_id')))
            .select(col('user_id').alias('node_id'), array_union(frontier['path'], array('user_id')).alias('path'))
        ).cache()

        updated_paths = (
            frontier
            .join(discovered_nodes, on='node_id', how='full_outer')
            .select('node_id',
                    when(
                        frontier['path'].isNotNull(), frontier['path']
                    ).otherwise(
                        discovered_nodes['path']
                    ).alias('path'))
        ).persist()

        if updated_paths.where(size(updated_paths['path']) == current_depth + 1).count() > 0:
            current_depth += 1
            frontier = discovered_nodes
        else:
            break

        if updated_paths.where(updated_paths['node_id'] == target_node).count() > 0:
            max_possible_depth = current_depth
            break

    result_paths = updated_paths.where(updated_paths['node_id'] == target_node).select('path').collect()
    formatted_paths = [','.join(map(str, path[0])) for path in result_paths]
    output_df = spark_session.createDataFrame([(line,) for line in formatted_paths], ['path_string'])
    output_df.write.mode('overwrite').text(output_file)
    spark_session.stop()

if __name__ == '__main__':
    main()