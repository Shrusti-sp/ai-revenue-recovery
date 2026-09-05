from pathlib import Path
import sqlite3
import pandas as pd

def database_path(project_root: Path) -> Path: return project_root / "data" / "database" / "revenue_recovery.db"
def write_database(frames: dict[str,pd.DataFrame], predictions: pd.DataFrame, project_root: Path):
    path=database_path(project_root); path.parent.mkdir(parents=True,exist_ok=True)
    with sqlite3.connect(path) as con:
        for name,df in frames.items(): df.to_sql(name,con,if_exists="replace",index=False)
        predictions.to_sql("predictions",con,if_exists="replace",index=False)
    return path
def query(project_root: Path, sql: str) -> pd.DataFrame:
    with sqlite3.connect(database_path(project_root)) as con: return pd.read_sql_query(sql,con)
