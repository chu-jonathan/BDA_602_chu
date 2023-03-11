import sys

from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCols, HasOutputCol
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, when
from pyspark.sql.window import Window


class RollingAverageTransformer(
    Transformer,
    HasInputCols,
    HasOutputCol,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    @keyword_only
    def __init__(self, inputCols=None, outputCol=None):
        super(RollingAverageTransformer, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)
        return

    @keyword_only
    def setParams(self, inputCols=None, outputCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset):
        input_cols = self.getInputCols()
        output_col = self.getOutputCol()

        # rolling average by window
        # https://stackoverflow.com/questions/45806194/pyspark-rolling-average-using-timeseries-data
        roll_w = Window.partitionBy("batter").orderBy("local_date").rowsBetween(-100, 0)
        rolling_average = (
            sum(col("hit")).over(roll_w)
            / sum(when(col("atbat") == 0, 0).otherwise(col("atbat"))).over(roll_w)
        ).alias(output_col)

        rolling_average_df = dataset.select(*input_cols, rolling_average)

        return rolling_average_df


def main():

    spark = SparkSession.builder.master("local[*]").getOrCreate()

    jdbc_driver = "org.mariadb.jdbc.Driver"
    batter_counts = (
        spark.read.format("jdbc")
        .option("url", "jdbc:mysql://localhost:3306/baseball?permitMysqlScheme")
        .option("dbtable", "baseball.batter_counts")
        .option("user", "jchu")
        .option("password", "bda")  # pragma: allowlist secret
        .option("driver", jdbc_driver)
        .load()
    )

    game = (
        spark.read.format("jdbc")
        .option("url", "jdbc:mysql://localhost:3306/baseball?permitMysqlScheme")
        .option("dbtable", "baseball.game")
        .option("user", "jchu")
        .option("password", "bda")  # pragma: allowlist secret
        .option("driver", jdbc_driver)
        .load()
    )

    batter_counts.createOrReplaceTempView("batter_counts")

    bc = spark.sql("SELECT game_id, batter, atbat, hit FROM batter_counts")

    game.createOrReplaceTempView("game")

    g = spark.sql("SELECT game_id, local_date FROM game")

    joined = bc.join(g, "game_id")

    joined.createOrReplaceTempView("joined")

    rolling_average_transformer = RollingAverageTransformer(
        inputCols=["batter", "local_date", "atbat", "hit"], outputCol="rolling_average"
    )

    rolling_average = rolling_average_transformer.transform(joined)
    rolling_average.show()


if __name__ == "__main__":
    sys.exit(main())
