import sys

from pyspark.sql import SparkSession


def main():

    spark = SparkSession.builder.master("local[*]").getOrCreate()

    jdbc_driver = "org.mariadb.jdbc.Driver"
    batter_counts = (
        spark.read.format("jdbc")
        .option("url", "jdbc:mysql://localhost:3306/baseball?permitMysqlScheme")
        .option("dbtable", "baseball.batter_counts")
        .option("user", "jchu")
        .option("password", "bda")
        .option("driver", jdbc_driver)
        .load()
    )

    game = (
        spark.read.format("jdbc")
        .option("url", "jdbc:mysql://localhost:3306/baseball?permitMysqlScheme")
        .option("dbtable", "baseball.game")
        .option("user", "jchu")
        .option("password", "bda")
        .option("driver", jdbc_driver)
        .load()
    )

    batter_counts.createOrReplaceTempView("batter_counts")
    bc = spark.sql("SELECT game_id, batter,atbat,hit FROM batter_counts")
    bc.show()
    game.createOrReplaceTempView("game")
    g = spark.sql("SELECT game_id, local_date FROM game")
    g.show()
    joined = bc.join(g, "game_id")
    joined.createOrReplaceTempView("joined")
    rolling_average = spark.sql(
        """
    SELECT a.batter, a.local_date,
        CASE
            WHEN SUM(b.atbat) > 0
            THEN SUM(b.hit) / SUM(b.atbat)
            ELSE 0
        END AS rolling_average
    FROM joined AS a
    JOIN joined AS b
        ON a.batter = b.batter
        AND b.local_date >= a.local_date - INTERVAL '100' DAY
        AND b.local_date < a.local_date
    GROUP BY a.batter, a.local_date
    LIMIT 10
"""
    )

    rolling_average.show()
    return


if __name__ == "__main__":
    sys.exit(main())
