import pickle  
import argparse  
import torch
import os
from utils.data_sampler import Data_Sampler
from utils import utils
import numpy as np

with open(f'dataset/VFF-1686demos', 'rb') as f:
    dataset = pickle.load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()  
    parser.add_argument('--device', default=0, type=int)                       # device, {"cpu", "cuda", "cuda:0", "cuda:1"}, etc
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--max_action", default=1., type=float)
    parser.add_argument("--num_epochs", default=1000, type=int)
    parser.add_argument("--num_steps_per_epoch", default=1000, type=int)
    parser.add_argument("--T", default=5, type=int)
    args = parser.parse_args()
    args.device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

    data_sampler = Data_Sampler(dataset, args.device)

    from agents.ql_diffusion import Diffusion_QL as Agent
    agent = Agent(state_dim=data_sampler.state_dim,
        action_dim=data_sampler.action_dim,
        max_action=args.max_action,
        device=args.device,
        discount=0.99,
        tau=0.005,
        max_q_backup=False,
        eta=0.,  # BC only
        n_timesteps=args.T,
        lr=args.lr,
        lr_decay=True,
        lr_maxt=args.num_epochs,
        grad_norm=1.0,
        )
    output_dir = 'models'
    model_idx = 2000

    agent.load_model(output_dir, model_idx)
    agent.model.eval()
    agent.actor.eval()

    # input concatenation of observation and goal, output action
    obs = dataset['observations'][1]
    goal = dataset['desired_goals'][1]
    true_action = dataset['actions'][1]
    state = np.concatenate([goal, obs])
    action = agent.sample_action(state)

    print(action, true_action)


