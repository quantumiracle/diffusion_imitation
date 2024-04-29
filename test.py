import pickle  
import argparse  
import torch
import os
from utils.data_sampler import Data_Sampler
from utils import utils
from torch.utils.tensorboard import SummaryWriter

with open(f'dataset/VFF-1686demos', 'rb') as f:
    dataset = pickle.load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()  
    parser.add_argument('--device', default=0, type=int)                       # device, {"cpu", "cuda", "cuda:0", "cuda:1"}, etc
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--max_action", default=1., type=float)
    parser.add_argument("--num_epochs", default=100, type=int)
    parser.add_argument("--num_steps_per_epoch", default=1000, type=int)
    parser.add_argument("--T", default=5, type=int)
    args = parser.parse_args()
    args.device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

    data_sampler = Data_Sampler(dataset, args.device)
    writer = SummaryWriter(f"runs/test")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

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
    output_dir = 'models/'
    os.makedirs(output_dir, exist_ok=True)
    training_iters = 0
    max_timesteps = args.num_epochs * args.num_steps_per_epoch
    while (training_iters < max_timesteps):
        loss_metric = agent.train(data_sampler,
                                  iterations=args.num_steps_per_epoch,
                                  batch_size=args.batch_size,
                                  log_writer=writer)       
        training_iters += args.num_steps_per_epoch
        curr_epoch = int(training_iters // int(args.num_steps_per_epoch))

        print(f"Training iterations: {training_iters}")
        agent.save_model(f"models/QL_diffusion_{training_iters}")
        utils.print_banner(f"Train step: {training_iters}", separator="*", num_star=90)

        if curr_epoch % 10 == 0:
            agent.save_model(output_dir, curr_epoch)

    writer.close()