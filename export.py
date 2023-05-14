import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import create_engine, text


def main():

    user = "jchu"
    password = "bda"  # pragma: allowlist secret
    host = "localhost"
    port = "3306"
    database = "baseball"

    engine = create_engine(
        f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}",
        future=True,
    )

    with engine.begin() as connection:
        features = connection.execute(
            text(
                "SELECT streak, win_lose, first_home_line, k_9, go_ao, bb_9, "
                "k_pitch_load, k_rest, month_column, days_since_last_game"
                "FROM pitching_stats"
            )
        )

    data = pd.DataFrame(features)
    export = pa.Table.from_pandas(data)
    pq.write_table(export, "export.parquet")


if __name__ == "__main__":
    sys.exit(main())
