from dataclasses import dataclass


@dataclass
class Config:
    n_block: int=5
    hidden_ch: int=128
    batch_size: int=32
    epoch: int=5
    db_path: str="Q.db"
    name: str="test"
    log_n: int=100
    eval_n: int=1000
    save_n: int=10000