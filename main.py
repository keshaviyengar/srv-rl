import gym
import argparse
from ddpg_her import DDPGAgent

import ctm2_envs

parser = argparse.ArgumentParser(description="Implementation of DDPG with HER with multiprocessing for "
                                             "concentric tube robots.")
#parser.add_argument("--env", type=str, default="bit-flipping-v0")
parser.add_argument("--env", type=str, default="Distal-1-Tube-Reach-v0")
parser.add_argument("--env-args", type=dict, default={})
parser.add_argument("--num-epochs", type=int, default=1000)
parser.add_argument("--num-episodes", type=int, default=20)
parser.add_argument("--buffer-size", type=int, default=5e5)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--gamma", type=int, default=0.98)
parser.add_argument("--actor-learning-rate", type=int, default=0.0001)
parser.add_argument("--critic-learning-rate", type=int, default=0.0001)
parser.add_argument("--tau", type=int, default=0.1)
parser.add_argument("--actor-mlp-units", type=int, default=256)
parser.add_argument("--actor-mlp-hidden-layers", type=int, default=1)
parser.add_argument("--critic-mlp-units", type=int, default=256)
parser.add_argument("--critic-mlp-hidden-layers", type=int, default=1)
parser.add_argument("--future-k", type=int, default=4)
parser.add_argument("--action-noise", type=int, default=1)
parser.add_argument("--training-json-file", type=str, default="training_parameters.json")
parser.add_argument("--hyperparam-constants-json-file", type=str, default="hyperparam_search_constants.json")

parser.add_argument("--train", type=bool, default=True)
parser.add_argument("--hyperparam-search", type=bool, default=False)

args = parser.parse_args()
args_dict = vars(args)
train = args.train
train_json_file = args.training_json_file
del args_dict["training_json_file"]
hyperparam_search = args.hyperparam_search
hyperparam_constants_json_file = args.hyperparam_constants_json_file
del args_dict["hyperparam_constants_json_file"]
print("training: %d hyperparameter-search: %d" % (train, hyperparam_search))
del args_dict["train"]
del args_dict["hyperparam_search"]

env = gym.make(args.env)
del args_dict["env_args"]
del args_dict["env"]

if train:
    rl_agent = DDPGAgent(env, **args_dict)
    rl_agent.train()
