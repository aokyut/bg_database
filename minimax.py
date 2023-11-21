import sqlite3
from PyBGEnv import qubic as env
# from PyBGEnv import connect4 as env
from tqdm import tqdm
from typing import Tuple, List
import random
from time import sleep
from math import log2
from argparse import ArgumentParser
from board_wrapper import show_qubic


def add(c: sqlite3.Cursor, b: str, val: int, depth: int):
    # print(b)
    c.execute("""
        insert or ignore into q_table
            values (?, ?, ?)
              """, (b, val, depth))

def delete(c: sqlite3.Cursor, b:str):
    c.execute("""
        delete from q_table where board=?
              """, (b,))


def search(c: sqlite3.Cursor, b: str):
    c.execute("""
        select val, level from q_table
        where board=?
            """, (b,))
    # print(c.fetchone())
    return c.fetchone()


def create(c):
    c.execute("""
        create table if not exists q_table (
            board text not null,
            val integer,
            level integer,
            primary key(board)
        )
    """)

init_hash = env.unique_hash(env.init())

def insert(c, board, val, level):
    b_hash = env.unique_hash(board)
    if init_hash == b_hash:
        assert False
    add(c, b_hash, val, level)


def peek(c, board, player, do_insert = True, depth = 1) -> Tuple[bool, Tuple[int, int, List[int]], int]:
    """
    Return:
        end_flag: bool
        result: -1, 0, 1
        action: int
    """
    # check in 1 turn
    # action = env.has_mate(board, player, 0)
    # if action > -1:
    #     if do_insert:
    #         insert(c, board, 1, 1)
    #     return True, (1, 1, [action]), action
    
    # check in 3 turn
    # action = env.has_mate(board, player, 2)
    # if action > -1:
    #     if do_insert:
    #         insert(c, board, 1, 3)
    #     return True, (1, 3, [action]), action
    action = env.has_mate(board, player, depth * 2)
    if action > -1:
        if do_insert:
            insert(c, board, 1, depth * 2 + 1)
        return True, (1, depth * 2 + 1, [action]), action
    
    
    best_action = -1
    max_val = -2
    min_level = 64
    action_cand = []
    if depth == 0:
        actions = env.valid_actions(board, player)
    else:
        actions = env.get_nonmate_actions(board, player, depth * 2 - 1)
    for action in actions:
    # for action in env.valid_actions(board, player):
        next_hash = env.unique_hash(env.get_next(board, action, player))
        row = search(c, next_hash)
        if row is None:
            action_cand.append(action)
            continue
        val = -row[0]
        level = row[1]
        if max_val < val:
            max_val = val
            min_level = level
            best_action = action
        elif max_val == val:
            if max_val == -1:
                if min_level < level:
                    min_level = level
                    best_action = action
            elif min_level > level:
                min_level = level
                best_action = action
    if max_val == 1:
        if do_insert:
            insert(c, board, max_val, min_level + 1)
        # TODO: max_val=1以外のものを削除する
        return True, (1, min_level+1, [best_action]), best_action
    elif len(action_cand) == 0:
        if max_val == 0:
            if do_insert:
                insert(c, board, 0, min_level + 1)
            return True, (0, min_level + 1, [best_action]), best_action
        elif max_val == -1:
            if do_insert:
                insert(c, board, -1, min_level + 1)
            return True, (-1, min_level + 1, [best_action]), best_action
        else:
            if do_insert:
                insert(c, board, 0, 0)
            return True, (0, 0, [-1]), -1
    else:
        return False, (0, 0, action_cand), random.choice(action_cand)


def board_search(c: sqlite3.Cursor, n: int, depth: int):
    b = env.init()
    player = 0
    entropy = []
    board_record = [b]
    step = 0
    min_step = 1000000
    bar = tqdm(range(n), smoothing=0.01, leave=False)
    ent_mean = Mean()
    step_mean = Mean()
    for _ in bar:
        while True:
            flag, (result, level, cand), action = peek(c, b, player, depth=depth)
            if flag:
                if player == 1:
                    result = -result
                message = ""
                if result == 1:
                    message = f"[step:{step:>3d}, black:{level:>2d}]"
                elif result == -1:
                    message = f"[step:{step:>3d}, white:{level:>2d}"
                else:
                    message = f"[step:{step:>3d},  draw:{level:>2d}"
                ent_mean.push(sum(entropy))
                step_mean.push(step)
                step -= 1
                if min_step > step:
                    min_step = step
                assert step <= 65

                bar.set_postfix(status=message, entropy=f"{ent_mean.mean:>6.4f}", min_step=min_step, mstep=f"{step_mean.mean:>6.4f}")
                player = 1 - player
                
                board_record = board_record[:-1]
                b = board_record[step]
                entropy = entropy[:-1]
                sleep(0.1)
                break
            entropy.append(log2(len(cand)))
            next_b = env.get_next(b, action, player)
            assert sum(next_b) > sum(b)
            b = next_b
            player = 1 - player
            board_record.append(b)
            step += 1
    return ent_mean.mean


class DatasetAgent:
    name = "DatasetAgent"
    def __init__(self, c):
        self.cur = c
        self.info = {}
    def get_action(self, board, player):
        flag, (result, level, cand), action = peek(c, board, player, False)
        self.info = {
            "flag": flag,
            "result": result,
            "level": level,
            "candidates": cand,
            "last_action": action
        }
        return action

class MinimaxAgent:
    name = "MinimaxAgent"
    def __init__(self, depth):
        self.depth = depth
        self.info = {}
    def get_action(self, board, player):
        for i in range(0, self.depth + 1, 2):
            action = env.has_mate(board, player, i)
            if action != -1:
                self.info = {
                    "flag": True,
                    "candidates": [action],
                    "last_action": action
                }
                return action
        
        if self.depth == 0:
            actions = env.valid_actions(board, player)
        else:
            actions = env.get_nonmate_actions(board, player, self.depth)
        action = random.choice(actions)
        self.info = {
            "flag": False,
            "candidates": actions,
            "last_action": action,
        }
        return action

class RandomAgent:
    name = "Random"
    def get_action(self, board, player):
        actions = env.valid_actions(board, player)
        action = random.choice(actions)
        self.info = {
            "candidates": actions,
            "last_action": action
        }
        return action

def play_agent(a1, a2):
    agents = [a1, a2]
    player = 0
    b = env.init()
    step = 0
    show_connect(b)
    while True:
        action = agents[player].get_action(b, player)
        if agents[player].info != None:
            print(agents[player].info)
        next_b = env.get_next(b, action, player)
        if env.is_done(next_b, player):
            show_connect(next_b)
            break
        b = next_b
        player = 1 - player
        step += 1
        show_connect(b)

def play_one(c) -> Tuple[str, float]:
    b = env.init()
    player = 0
    step = 0
    entropy = 0
    while True:
        step += 1
        flag, (result, level, cand), action = peek(c, b, player)
        entropy += log2(len(cand))
        if flag:
            if player == 1:
                result = -result
            if result == 1:
                return f"[step:{step:>3d}, black:{level:>2d}]", entropy
            elif result == -1:
                return f"[step:{step:>3d}, white:{level:>2d}]", entropy
            else:
                return f"[step:{step:>3d},  draw:{level:>2d}]", entropy
        b = env.get_next(b, action, player)
        player = 1 - player


class Mean:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
    def push(self, val: float):
        self.mean = self.mean * (self.n / (self.n + 1)) + val * (1 / (self.n + 1))
        self.n += 1


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

def test(c):
    b = env.init()
    player = 0
    while True:
        result = peek(c, b, player)
        print(result)
        _, _, action = result
        next_b = env.get_next(b, action, player)
        b = next_b
        show_connect(b)
        if env.is_done(b, player):
            print("end")
            break
        player = 1 - player

    assert False


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("file", type=str)
    parser.add_argument("--depth", type=int)

    args = parser.parse_args()
    conn = sqlite3.connect(args.file)
    c = conn.cursor()
    create(c)
    
    # a1 = DatasetAgent(c)
    # a2 = DatasetAgent(c)
    # a2 = MinimaxAgent(3)
    # a2 = RandomAgent()
    # play_agent(a1, a2)
    # exit(0)

    bar = tqdm(range(10000), smoothing=0.01)
    entropy = Mean()
    for i in bar:
        sleep(0.05)

        ent = board_search(c, 100, args.depth)
        # entropy.push(ent)
        conn.commit()
    
        res, ent = play_one(c)
        # print(res, ent)
        entropy.push(ent)
        with open("out", "a") as f:
            print(ent, file=f)
        bar.set_postfix(status=res, entropy_mean=f"{entropy.mean:>6.4f}", entropy=ent)
        # bar.set_postfix(entropy=f"{entropy.mean:>6.4f}")

    conn.commit()