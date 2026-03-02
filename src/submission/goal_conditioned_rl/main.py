"""Main script to setup goal-conditioned RL experiments."""
import argparse

import numpy as np
import torch
from torch.utils import tensorboard

from .bit_flip_env import BitFlipEnv
import gymnasium as gym
import metaworld
from .sawyer_action_discretize import SawyerActionDiscretize

from .trainer import train
from .utils import HERType


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Flip some bits!')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='directory to save to or load from')
    parser.add_argument('--env', type=str, default='bit_flip',
                        help='choose between bit_flip_env and sawyer_reach')
    parser.add_argument('--num_bits', type=int, default=7,
                        help='number of bits in the bit flipping environment')
    parser.add_argument('--num_epochs', type=int, default=250,
                        help='number of epochs Q-learning is run for')
    parser.add_argument('--her_type', type=str, default='no_hindsight',
                        help='type of HER to use')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='manual seed for pytorch')
    args = parser.parse_args()

    torch.manual_seed(args.random_seed)

    if args.env == 'bit_flip':
        env = BitFlipEnv(args.num_bits)
        input_dim = 2 * args.num_bits
        action_dim = args.num_bits
        steps_per_episode = args.num_bits
        env_reward_function = lambda x, y: 0.0 if np.array_equal(x, y) else -1.0
        tensorboard_log_dir = (
            f'./goal_conditioned_rl/logs/gcrl/bit_flip/num_bits:{args.num_bits}/HER_type:{args.her_type}/seed:{args.random_seed}' # pylint: disable=line-too-long
        ) if args.log_dir is None else args.log_dir

    elif args.env == 'sawyer_reach':
        env = gym.make("Meta-World/MT1", env_name="reach-v3", seed=args.random_seed)
        env = SawyerActionDiscretize(
            env, render_every_step=False)
        input_dim = 4
        action_dim = 4
        steps_per_episode = 50
        env_reward_function = lambda x, y: -np.linalg.norm(x - y)
        tensorboard_log_dir = (
            f'./goal_conditioned_rl/logs/gcrl/sawyer_reach/HER_type:{args.her_type}/seed:{args.random_seed}' # pylint: disable=line-too-long
        ) if args.log_dir is None else args.log_dir

    print(f'logging experiment at: {tensorboard_log_dir}')
    writer = tensorboard.SummaryWriter(log_dir=tensorboard_log_dir)
    train(
        env=env,
        input_dim=input_dim,
        action_dim=action_dim,
        env_reward_function=env_reward_function,
        num_epochs=args.num_epochs,
        steps_per_episode=steps_per_episode,
        her_type=HERType[args.her_type.upper()],
        writer=writer,
    )
