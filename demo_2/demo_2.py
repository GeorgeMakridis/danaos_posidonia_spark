import datetime
from pyspark.sql import SparkSession
from demo_2 import DetectorLevel_1

spark = SparkSession.builder.appName('ml-danaos').master("local[*]").config("spark.driver.memory", "12g") \
    .config('spark.storage.memoryFraction', '0').config('spark.driver.maxResultSize', '12g').config(
    'spark.kryoserializer.buffer.max', '1999m') \
    .config('spark.debug.maxToStringFields', '100').getOrCreate()
print(spark.sparkContext._conf.getAll())

print(datetime.datetime.now())
detector_1 = DetectorLevel_1.DetectorLevel_1(param='message', sc=spark)
detector_1.run_xgb_training('XGB prediction started..')
