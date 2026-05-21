import os
import csv
from pathlib import Path
import pandas as pd

from dotenv import load_dotenv
import psycopg
from psycopg import sql


load_dotenv()

POSTGRES_CONN_STRING = os.getenv("POSTGRES_CONN_STRING")
FIRST_CSV_PATH = "./output_data/prep_logs/history_consecutive.csv"  # used to infer table name and columns
CSV_FILE_PATHS = [
    "./output_data/topic/topic_consecutive.csv",
    "./output_data/mental/mental_consecutive.csv",
    "./output_data/sentiment/sentiment_consecutive.csv"
]  # list of input CSV file paths

TARGET_TABLE_NAME = "londoners_chat_history"  # target table name in PostgreSQL

if not POSTGRES_CONN_STRING:
    raise ValueError("POSTGRES_CONN_STRING not found in .env")


def infer_table_name(csv_path: Path) -> str:
    # file customers.csv -> table customers
    return csv_path.stem.lower().replace(" ", "_").replace("-", "_")


def read_csv_header(csv_path: Path) -> list[str]:
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
    return [col.strip().lower().replace(" ", "_").replace("-", "_") for col in header]


def create_table_if_not_exists(conn, table_name: str, columns: list[str]) -> None:
    # Creates all columns as TEXT by default.
    # You can later change types manually if needed.
    cols_sql = sql.SQL(", ").join(
        sql.SQL("{} TEXT").format(sql.Identifier(col)) for col in columns
    )

    query = sql.SQL(
        "CREATE TABLE IF NOT EXISTS {} ({})"
    ).format(
        sql.Identifier(table_name),
        cols_sql
    )

    with conn.cursor() as cur:
        cur.execute(query)
    conn.commit()


def truncate_table(conn, table_name: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("TRUNCATE TABLE {}").format(sql.Identifier(table_name))
        )
    conn.commit()


def load_csv_to_table(conn, csv_path: Path, table_name: str, columns: list[str]) -> None:
    copy_query = sql.SQL(
        "COPY {} ({}) FROM STDIN WITH (FORMAT CSV, HEADER TRUE)"
    ).format(
        sql.Identifier(table_name),
        sql.SQL(", ").join(sql.Identifier(col) for col in columns)
    )

    with conn.cursor() as cur:
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            with cur.copy(copy_query) as copy:
                while data := f.read(8192):
                    copy.write(data)
    conn.commit()

def merge_csv_files(
        first_csv_path: Path,
        csv_paths: list[Path], 
        target_table_name: str,
        ) -> Path:
    # This function is optional and can be used to merge multiple CSVs into one before loading.
    # It assumes all CSVs have the same structure (same columns).

    first_df = pd.read_csv(first_csv_path)

    common_cols = []
    for path in csv_paths:
        df = pd.read_csv(path)
        if df.empty:
            continue
        common_cols.extend([col for col in df.columns if col in first_df.columns])
    
    common_cols = list(set(common_cols))
    
    print(f"Common columns for merging: {common_cols}")

    df_tot = first_df.copy()
    for path in csv_paths:
        df = pd.read_csv(path)
        df_tot = pd.merge(df_tot, df, on=common_cols, how="outer")

    merged_path = Path(f"./output_data/{target_table_name}.csv")
    df_tot.to_csv(merged_path, index=False)

    return merged_path


def main():
    csv_files = [Path(path_str) for path_str in CSV_FILE_PATHS]

    missing_files = [str(path) for path in csv_files if not path.is_file()]
    if missing_files:
        print("These CSV files were not found:")
        for path in missing_files:
            print(f"- {path}")
        return

    merged_path = merge_csv_files(first_csv_path=FIRST_CSV_PATH, csv_paths=csv_files, target_table_name=TARGET_TABLE_NAME)

    with psycopg.connect(POSTGRES_CONN_STRING) as conn:

        table_name = infer_table_name(merged_path)
        columns = read_csv_header(merged_path)

        print(f"Processing {merged_path.name} -> table {table_name}")

        create_table_if_not_exists(conn, table_name, columns)

        # optional: clear table before loading fresh data
        truncate_table(conn, table_name)

        load_csv_to_table(conn, merged_path, table_name, columns)

        print(f"Loaded {merged_path.name} into {table_name}")

        # for csv_path in csv_files:
        #     table_name = infer_table_name(csv_path)
        #     columns = read_csv_header(csv_path)

        #     print(f"Processing {csv_path.name} -> table {table_name}")

        #     create_table_if_not_exists(conn, table_name, columns)

        #     # optional: clear table before loading fresh data
        #     truncate_table(conn, table_name)

        #     load_csv_to_table(conn, csv_path, table_name, columns)

        #     print(f"Loaded {csv_path.name} into {table_name}")

    print("Done.")


if __name__ == "__main__":
    main()
