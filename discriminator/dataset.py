import sqlite3
from PyBGEnv import qubic
import numpy as np
import random
from .config import Config

def decode_qubic(b_hash: str):
    idx = 0
    b = qubic.init()
    for c in b_hash:
        code = ord(c)
        for _ in range(8):
            if code & 1 == 1:
                b[idx] = 1
            idx += 1
    return b

rotate16 = [3, 7, 11, 15, 
            2, 6, 10, 14, 
            1, 5,  9, 13,
            0, 4,  8, 12]
rotate64 = [r + i * 16 for r in rotate16 for i in range(8)]

flip16 = [
    3, 2, 1, 0,
    7, 6, 5, 4,
    11, 10, 9, 8,
    15, 14, 13, 12
]
flip64 = [r + i * 16 for r in flip16 for i in range(8)]

class Dataset:
    def __init__(self, config:Config):
        self.conn = sqlite3.connect(config.db_path)
        self.cur = self.conn.cursor()
        self.size = self.get_size()
        self.batch_size = config.batch_size
    
    def get_size(self) -> int:
        self.cur.execute("select count(*) from q_table")
        return self.cur.fetchone()[0]
    
    def make_one_batch(self, board_hash:str, val: int):
        """
        Return:
            board: numpy[128]
            val_label: numpy[3](one-hot)
        """
        board = decode_qubic(board_hash)
        val_label = np.array([val + 1])
        r = random.random()
        if r < 0.125:
            return board, val_label
        elif r < 0.25:
            return board[flip64], val_label
        board = board[rotate64]
        if r < 0.375:
            return board, val_label
        elif r < 0.5:
            return board[flip64], val_label
        board = board[rotate64]
        if r < 0.625:
            return board, val_label
        elif r < 0.75:
            return board[flip64], val_label
        board = board[rotate64]
        if r < 0.875:
            return board, val_label
        return board[flip64], val_label
    
    def generate_batch(self):
        self.cur.execute("""
            select board, val from q_table order by random()
        """)
        size = 0
        boards = []
        vals = []
        while True:
            res = self.cur.fetchone()
            if res is None:
                break
            board_hash, val = res
            try:
                board, val_label = self.make_one_batch(board_hash, val)
            except:
                continue
            size += 1
            boards.append(board)
            vals.append(val_label)
            if size < self.batch_size:
                continue
            batch_board = np.vstack(boards)
            batch_val = np.vstack(vals)
            boards = []
            vals = []
            yield batch_board, batch_val

