import sqlite3 
from argparse import ArgumentParser


def merge(tarpath: str, srcpath: str):
    conn = sqlite3.connect(tarpath)
    c = conn.cursor()
    c.execute(f"attach database \"{srcpath}\" as src")
    
    # insert src.q_table into tar.q_table
    c.execute("""
        insert into q_table(board, val, level)
            select board, val, level from src.q_table
        where true
        on conflict(board)
            do update set level = excluded.level
            where (val = 1 and level > excluded.level) or (val = -1 and level < excluded.level)
              """)
    
    conn.commit()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("tar_path", type=str)
    parser.add_argument("src_path", type=str)
    args = parser.parse_args()

    merge(args.tar_path, args.src_path)
