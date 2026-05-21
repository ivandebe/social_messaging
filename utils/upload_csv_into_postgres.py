import os
import csv
from pathlib import Path
from dotenv import load_dotenv
import psycopg
from psycopg import sql

load_dotenv()

POSTGRES_CONN_STRING = os.getenv("POSTGRES_CONN_STRING")
CSV_FILE_PATHS = [
    "./output_data/prep_logs/history_consecutive.csv",
    "./output_data/topic/topic_consecutive.csv",
    "./output_data/mental/mental_consecutive.csv",
    "./output_data/sentiment/sentiment_consecutive.csv"
]  # list of input CSV file paths

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


def main():
    csv_files = [Path(path_str) for path_str in CSV_FILE_PATHS]

    missing_files = [str(path) for path in csv_files if not path.is_file()]
    if missing_files:
        print("These CSV files were not found:")
        for path in missing_files:
            print(f"- {path}")
        return

    with psycopg.connect(POSTGRES_CONN_STRING) as conn:
        for csv_path in csv_files:
            table_name = infer_table_name(csv_path)
            columns = read_csv_header(csv_path)

            print(f"Processing {csv_path.name} -> table {table_name}")

            create_table_if_not_exists(conn, table_name, columns)

            # optional: clear table before loading fresh data
            truncate_table(conn, table_name)

            load_csv_to_table(conn, csv_path, table_name, columns)

            print(f"Loaded {csv_path.name} into {table_name}")

    print("Done.")


if __name__ == "__main__":
    main()
