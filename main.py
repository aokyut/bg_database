from discriminator.config import Config
from discriminator.network import Discriminator
from discriminator.dataset import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter
from PyBGEnv import qubic as env
import numpy as np
import random
from tqdm import tqdm
from board_wrapper import show_qubic



class ModelAgent:
    def __init__(self, model: Discriminator):
        self.model = model
    
    def get_action(self, state, player):
        valid_actions = env.valid_actions(state, player)
        next_states = []
        for action in valid_actions:
            next_states.append(env.get_next(state, action, player))
        next_states = torch.FloatTensor(np.vstack(next_states))
        values = torch.sum(model(next_states) * torch.tensor([[-1, 0, 1]]), dim=-1)
        idx = torch.argmin(values).item()
        return valid_actions[idx]

class Minimax:
    def __init__(self, depth:int):
        self.depth = depth
        self.name = f"Minimax:{depth}"
    
    def get_action(self, state, player):
        return env.minimax_action(state, player, self.depth)


class Random:
    def __init__(self):
        self.name = "Random"
        pass

    def get_action(self, state, player):
        valid_action = env.valid_actions(state, player)
        return random.choice(valid_action)


def player_agents(a1, a2):
    agents = [a1, a2]
    b = env.init()
    player = 0
    while not env.is_done(b, 1 - player):
        action = agents[player].get_action(b, player)
        b = env.get_next(b, action, player)
        player = 1- player
    if env.is_draw(b):
        return 0.5, 0.5
    if 1 - player == 0:
        return 1, 0
    else:
        return 0, 1
    

def eval_func(model: Discriminator):
    eval_agents = [
        Random(),
        Minimax(0),
        Minimax(1),
        Minimax(2)
    ]
    tar_agent = ModelAgent(model)
    eval_num = 10
    score = {}
    for agent in tqdm(eval_agents, leave=False):
        result = 0
        for _ in tqdm(range(eval_num // 2), leave=False, desc=f"{agent.name}:black"):
            res, _ = player_agents(tar_agent, agent)
            result += res
        for _ in tqdm(range(eval_num // 2), leave=False, desc=f"{agent.name}:white"):
            _, res = player_agents(agent, tar_agent)
            result += res
        score[agent.name] = result / eval_num
    return score


def train(model: Discriminator, dataset: Dataset, config: Config):
    model_optim = optim.Adam(model.parameters())
    writer = SummaryWriter(f"tensorboard/{config.name}")
    epoch = config.epoch

    bar_epoch = tqdm(range(epoch), desc="[Epoch]")
    step = 0
    for _ in bar_epoch:
        bar_batch = tqdm(dataset.generate_batch(), smoothing=0.01, leave=False, total=dataset.get_size() // config.batch_size)
        for board, label in bar_batch:
            board = torch.FloatTensor(board)
            label = torch.reshape(torch.LongTensor(label), shape=(-1,))
            
            predict = model.value(board)
            
            loss = F.cross_entropy(predict, label)

            model_optim.zero_grad()
            loss.backward()
            model_optim.step()

            if step % config.log_n == 0:
                writer.add_scalar("loss/cross_entropy", loss.item(), step)
            if step % config.eval_n == 0:
                result = eval_func(model)
                for key, value in result.items():
                    writer.add_scalar(f"eval/{key}", value, step)
            if step % config.save_n == 0:
                torch.save(model.state_dict(), "discriminator.pth")
            
            bar_batch.set_postfix(loss=f"{loss.item():2.4f}")
    
            step += 1


if __name__ == "__main__":
    config = Config(
        n_block=5,
        hidden_ch=128,
        batch_size=32,
        epoch=5,
        db_path="Q.db",
        name="test"
    )

    model = Discriminator(config)
    dataset = Dataset(config)

    train(model, dataset, config)