import os
from dotenv import load_dotenv
import psycopg
from psycopg import sql
from psycopg.rows import dict_row
import pandas as pd

load_dotenv()


def get_db_connection():
    conn_string = os.getenv("POSTGRES_CONN_STRING")
    if not conn_string:
        raise ValueError("POSTGRES_CONN_STRING not found in .env")
    return psycopg.connect(conn_string, row_factory=dict_row)


def fetch_entire_table(table_name: str) -> pd.DataFrame:
    query = sql.SQL("SELECT * FROM {}").format(sql.Identifier(table_name))

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    return df
