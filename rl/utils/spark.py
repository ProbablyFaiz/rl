import pyspark

import rl.utils.io
from rl.utils import LOGGER


def get_spark_session() -> pyspark.sql.SparkSession:
    rl.utils.io.ensure_dotenv_loaded()
    return (
        pyspark.sql.SparkSession.builder.appName("cdle")
        # .master(rl.utils.io.getenv("SPARK_URL", "local[*]"))
        .getOrCreate()
    )


def get_spark() -> tuple[pyspark.sql.SparkSession, pyspark.SparkContext]:
    spark = get_spark_session()
    sc = spark.sparkContext
    # LOGGER.warn("Building the project egg and adding it to the Spark context...")
    # sc.addArchive(str(rl.utils.io.create_project_egg()))
    sc.setSystemProperty("spark.executor.memory", "4g")
    sc.setSystemProperty("spark.executor.instances", "2")
    return spark, sc


def stop_spark() -> None:
    get_spark_session().stop()
