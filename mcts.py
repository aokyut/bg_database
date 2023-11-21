import sqlite3
from PyBGEnv import qubic as env
# from PyBGEnv import connect4 as env
from board_wrapper import decode_qubic, show_qubic
from tqdm import tqdm
from typing import Tuple, List
import random
from time import sleep
import math
from argparse import ArgumentParser

def show_connect(b):
    s = ""
    for i in range(5, -1, -1):
        for j in range(7):
            if b[i * 7 + j] == 1:
                s += "O"
            elif b[i * 7 + j + 42] == 1:
                s += "X"
            else:
                s += "-"
        s += "\n"
    print(s)

def create(c):
    c.execute("""
        create table if not exists mct (
            board text not null,
            action integer not null check (typeof(action) = 'integer'),
            n integer not null,
            win integer not null,
            primary key (board, action)
        )
        """)
    c.execute("PRAGMA table_info(mct)")
    print(c.fetchall())
    c.execute("PRAGMA strict=true")

def lookup(c: sqlite3.Cursor, b: str):
    c.execute("""
        select action, n, win from mct
        where board=?
    """, (b,))
    return c.fetchall()

def push_many(c: sqlite3.Cursor, params):
    c.executemany("""
        insert into mct(board, action, n, win)
            values (?, ?, ?, ?)
        on conflict(board, action)
        do update 
            set
            n = excluded.n,
            win = excluded.n
                  """, params)

def update_one(c: sqlite3.Cursor, board, action, n, win):
    c.execute("""
        update mct
        set
            n=?, win=?
        where
            board=? and action=?
              """, (n, win, board, action))

def update_many(c: sqlite3.Cursor, params):
    c.executemany("""
        update mct
        set
            n=?, win=?
        where
            board=? and action=?
              """, params)


def mcts(c: sqlite3.Cursor, sim_n:int, expand_n:int, uct_c: float, b, player, display=False) -> int:
    rows = lookup(c, env.unique_hash(b))
    if len(rows) == 0:
        expand(c, b, player)
    bar = tqdm(range(sim_n), desc="[mcts]", leave=False, smoothing=0.01)
    for i in bar:
        _, update_query = evaluate(c, expand_n, uct_c, b, player)
        update_many(c, update_query)
        assert True
    query = lookup(c, env.unique_hash(b))
    if display:
        for que in query:
            print(f"[{que[0]:>3d}]: {que[2]:>6d}/{que[1]:>6d} ({100 * que[2] / que[1]:>3.1f}%)")
    max_action = max(query, key=lambda x: x[1])
    return max_action[0]


def evaluate(cur: sqlite3.Cursor, expand_n:int, uct_c: float, b, player) -> Tuple[int, List[Tuple[str, int, int, int]]]:
    query = lookup(cur, env.unique_hash(b))

    if len(query) == 0:
        result = playout(b, player)
        return result, []
    else:
        # ucts = []
        N = sum([q[1] for q in query])
        # for action, n, win in query:
        #     ucts.append((action, n, win, win / n + uct_c * math.sqrt(math.log(N)/n)))
        max_action = max(query, key=lambda x: x[2] / x[1] + uct_c * math.sqrt(math.log(N)/x[1]))
        next_b = env.get_next(b, max_action[0], player)
        new_n = max_action[1] + 1
        if new_n == expand_n and not env.is_done(next_b, player):
            expand(cur, next_b, 1 - player)
        result, update_query = evaluate(cur, expand_n, uct_c, next_b, 1 - player)
        new_win = max_action[2]
        if result == -1:
            new_win += 1
        update_query.append(
            (new_n, new_win, env.unique_hash(b), max_action[0])
        )
        return -result, update_query

def playout(board, player) -> int:
    coef = 1
    while not env.is_done(board, 1 - player):
        action = random.choice(env.valid_actions(board, player))
        next_board = env.get_next(board, action, player)
        if env.is_win(next_board, player):
            return coef
        if env.is_draw(next_board):
            break
        coef *= -1
        board = next_board
        player = 1 - player
    if env.is_win(board, 1 - player):
        return 1
    return 0


def expand(c: sqlite3.Cursor, b, player):
    # actions = env.get_nonmate_actions(b, player, 3)
    actions = env.valid_actions(b, player)
    params = []
    for action in actions:
        params.append((env.unique_hash(b), int(action), 1, 0))
    push_many(c, params)
    # print(lookup(c, env.unique_hash(b)))


def play(c, expand_n, uct_c, sim_n):
    b = env.init()
    player = 0
    while not env.is_done(b, 1 - player):
        action = mcts(c, sim_n, expand_n, uct_c, b, player)
        b = env.get_next(b, action, player)
        player = 1 - player
        sleep(1)
        # break

class DBAgent:
    def __init__(self, c:sqlite3.Cursor, sim_num:int = 1000, uct_c: float=math.sqrt(2), expand_n=100):
        self.c = c
        self.sim_num = sim_num
        self.uct_c = uct_c
        self.expand_n = expand_n
    def get_action(self, board, player):
        action = mcts(self.c, self.sim_num, self.expand_n, self.uct_c, board, player)
        return action

class Minimax:
    def __init__(self, depth:int=1):
        self.depth = 1
    def get_action(self, board, player):
        action = env.minimax_action(board, player, self.depth)
        return action

def eval_agents(a1, a2):
    agents = [a1, a2]
    b = env.init()
    player = 0
    while not env.is_done(b, 1 - player):
        action = agents[player].get_action(b, player)
        next_b = env.get_next(b, action, player)
        b = next_b
        player = 1 - player
    result = [0, 0]
    if env.is_win(b, 1 - player):
        result[1 - player] = 1
        result[player] = -1
    return result


def selfplay(args):
    conn = sqlite3.connect(args.db_path)
    c = conn.cursor()
    create(c)
    bar = tqdm(range(args.num_selfplay), smoothing=0.01, desc="[selfplay]")
    for i in bar:
        play(c, args.n_expand, math.sqrt(2), args.n_sim)
        conn.commit()
        
    return

def decode_connect(b_hash: str):
    b = env.init()
    idx = 0
    for c in b_hash:
        code = ord(c)
        for _ in range(7):
            if (code & 1) == 1:
                b[idx] = 1
            idx += 1
            code >>= 1
    return b


def treewalk(args):
    conn = sqlite3.connect(args.db_path)
    c = conn.cursor()
    create(c)
    b = env.init()
    boards = [b]
    while True:
        describe(c, boards[-1])
        insts = input(">>").strip().split()
        if insts[0] == "up":
            boards = up(boards)
        elif insts[0] == "down":
            boards = down(boards, insts[1:])
        elif insts[0] == "search":
            search(c, insts)


RED = "\033[31m"
GREEN = "\033[32m"
BLUE = "\033[34m"
END = "\033[0m"

def describe(c: sqlite3.Cursor, b):
    rows = lookup(c, env.unique_hash(b))
    if len(rows) == 0:
        show_qubic(b)
        print("row is none")
        return
    max_row = max(rows, key=lambda x: x[1])
    min_row = max(rows, key=lambda x: -x[1])
    max_action = max_row[0]
    min_action = min_row[0]
    # show_connect(b)
    show_qubic(b)
    for row in rows:
        if row[0] == max_action:
            print(GREEN + f"[{row[0]:>3d}]: {row[2]:>6d}/{row[1]:>6d} ({100 * row[2] / row[1]:>3.1f}%)" + END)
        elif row[0] == min_action:
            print(RED + f"[{row[0]:>3d}]: {row[2]:>6d}/{row[1]:>6d} ({100 * row[2] / row[1]:>3.1f}%)" + END)
        else:
            print(f"[{row[0]:>3d}]: {row[2]:>6d}/{row[1]:>6d} ({100 * row[2] / row[1]:>3.1f}%)")

def up(boards):
    if len(boards) == 1:
        print("Now on root")
        return boards
    else:
        return boards[:-1]

def down(boards, insts):
    try:
        action = int(insts[0])
        board = boards[-1]
        player = env.current_player(board)
        if not action in env.valid_actions(board, player):
            print(f"action:{action} is invald.")
        next_board = env.get_next(board, action, player)
        boards.append(next_board)
        return boards
    except Exception as e:
        print(e)
        return boards

def search(c: sqlite3.Cursor, insts, board):
    pass

if __name__ == "__main__":
    conn = sqlite3.connect("mcts.db")
    c = conn.cursor()
    create(c)

    # play(c, 100, math.sqrt(2), 1000)
    
    conn.commit()
    parser = ArgumentParser(description="mcts dataset operator")
    subparsers = parser.add_subparsers()

    parser_selfplay = subparsers.add_parser("selfplay", help="see `selfplay -h`")
    parser_selfplay.add_argument("db_path")
    parser_selfplay.add_argument("-n", "--num_selfplay", type=int, default=1000)
    parser_selfplay.add_argument("--n_sim", type=int, default=1000)
    parser_selfplay.add_argument("--n_expand", type=int, default=100)
    parser_selfplay.add_argument("--n_eval", type=int, default=100)
    parser_selfplay.set_defaults(handler=selfplay)

    parser_walk = subparsers.add_parser("walk", help="see `walk -h`")
    parser_walk.add_argument("db_path")
    parser_walk.add_argument("--n_sim", type=int, default=1000)
    parser_walk.add_argument("--n_expand", type=int, default=100)
    parser_walk.set_defaults(handler=treewalk)

    args = parser.parse_args()
    if hasattr(args, 'handler'):
        args.handler(args)
    else:
        # 未知のサブコマンドの場合はヘルプを表示
        parser.print_help()