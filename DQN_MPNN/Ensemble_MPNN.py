# -*- coding: utf-8 -*-
from __future__ import division
import argparse
import bz2
from datetime import datetime
import gym
import pickle
import atari_py
import gym_environments
from tqdm import trange
from DQN_MPNN.DQNagent import DQNAgent as Agent
from DQN_MPNN.Ensemble_memory import ReplayMemory
from Ensemble.test import *
from Ensemble.state_feature import *
from DQN_MPNN.DQN_config import *

# Environment
graph_topology = 0 # 0==NSFNET, 1==GEANT2, 2==Small Topology, 3==GBN
SEED = 37
listofDemands = [8, 32, 64]
env = gym.make('GraphEnv-v1')
np.random.seed(SEED)
env.seed(SEED)
env.generate_environment(graph_topology, listofDemands)

env_eval = gym.make('GraphEnv-v1')
np.random.seed(SEED)
env_eval.seed(SEED)
env_eval.generate_environment(graph_topology, listofDemands)

state_size = env.numEdges
ACTION_LEN = 4

# Note that hyperparameters may originally be reported in ATARI game frames instead of agent steps
parser = argparse.ArgumentParser(description='Rainbow')
parser.add_argument('--id', type=str, default='boot_rainbow', help='Experiment ID')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--game', type=str, default='space_invaders', choices=atari_py.list_games(), help='ATARI game')
parser.add_argument('--T-max', type=int, default=int(1e5), metavar='STEPS',
                    help='Number of training steps (4x number of frames)')
parser.add_argument('--max-episode-length', type=int, default=int(108e3), metavar='LENGTH',
                    help='Max episode length in game frames (0 to disable)')
parser.add_argument('--history-length', type=int, default=1, metavar='T', help='Number of consecutive states processed')
parser.add_argument('--architecture', type=str, default='data-efficient', choices=['canonical', 'data-efficient'],
                    metavar='ARCH', help='Network architecture')
parser.add_argument('--hidden-size', type=int, default=256, metavar='SIZE', help='Network hidden size')
parser.add_argument('--noisy-std', type=float, default=0.1, metavar='σ',
                    help='Initial standard deviation of noisy linear layers')
parser.add_argument('--atoms', type=int, default=51, metavar='C', help='Discretised size of value distribution')
parser.add_argument('--V-min', type=float, default=-10, metavar='V', help='Minimum of value distribution support')
parser.add_argument('--V-max', type=float, default=10, metavar='V', help='Maximum of value distribution support')
parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
parser.add_argument('--memory-capacity', type=int, default=int(500000), metavar='CAPACITY',
                    help='Experience replay memory capacity')
parser.add_argument('--replay-frequency', type=int, default=50, metavar='k', help='Frequency of sampling from memory')
parser.add_argument('--priority-exponent', type=float, default=0.5, metavar='ω',
                    help='Prioritised experience replay exponent (originally denoted α)')
parser.add_argument('--priority-weight', type=float, default=0.4, metavar='β',
                    help='Initial prioritised experience replay importance sampling weight')
parser.add_argument('--multi-step', type=int, default=20, metavar='n', help='Number of steps for multi-step return')
parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
parser.add_argument('--target-update', type=int, default=int(2000), metavar='τ',
                    help='Number of steps after which to update target network')
parser.add_argument('--reward-clip', type=int, default=1, metavar='VALUE', help='Reward clipping (0 to disable)')
parser.add_argument('--learning-rate', type=float, default=0.0001, metavar='η', help='Learning rate')
parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='ε', help='Adam epsilon')
parser.add_argument('--batch-size', type=int, default=32, metavar='SIZE', help='Batch size')
parser.add_argument('--learn-start', type=int, default=int(1600), metavar='STEPS',
                    help='Number of steps before starting training')
parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
parser.add_argument('--evaluation-interval', type=int, default=TRAINING_EPISODES, metavar='STEPS',
                    help='Number of training steps between evaluations')
parser.add_argument('--evaluation-episodes', type=int, default=EVALUATION_EPISODES, metavar='N',
                    help='Number of evaluation episodes to average over')
# TODO: Note that DeepMind's evaluation method is running the latest agent for 500K frames ever every 1M steps
parser.add_argument('--evaluation-size', type=int, default=40, metavar='N',
                    help='Number of transitions to use for validating Q')
parser.add_argument('--render', action='store_true', help='Display screen (testing only)')
parser.add_argument('--enable-cudnn', action='store_true', help='Enable cuDNN (faster but nondeterministic)')
parser.add_argument('--memory', help='Path to save/load the memory from')
parser.add_argument('--disable-bzip-memory', action='store_true',
                    help='Don\'t zip the memory file. Not recommended (zipping is a bit slower and much, much smaller)')
# ensemble
parser.add_argument('--num-ensemble', type=int, default=5, metavar='N', help='Number of ensembles')
parser.add_argument('--beta-mean', type=float, default=1, help='mean of bernoulli')
parser.add_argument('--temperature', type=float, default=10, help='temperature for CF')
parser.add_argument('--ucb-infer', type=float, default=1.0, help='coeff for UCB infer')
parser.add_argument('--ucb-train', type=float, default=0.0, help='coeff for UCB train')


# Simple ISO 8601 timestamped logger
def log(s):
    print('[' + str(datetime.now().strftime('%Y-%m-%dT%H:%M:%S')) + '] ' + s)


def load_memory(memory_path, disable_bzip):
    if disable_bzip:
        with open(memory_path, 'rb') as pickle_file:
            return pickle.load(pickle_file)
    else:
        with bz2.open(memory_path, 'rb') as zipped_pickle_file:
            return pickle.load(zipped_pickle_file)


def save_memory(memory, memory_path, disable_bzip):
    if disable_bzip:
        with open(memory_path, 'wb') as pickle_file:
            pickle.dump(memory, pickle_file)
    else:
        with bz2.open(memory_path, 'wb') as zipped_pickle_file:
            pickle.dump(memory, zipped_pickle_file)


# Setup
args = parser.parse_args()
print(' ' * 26 + 'Options')
for k, v in vars(args).items():
    print(' ' * 26 + k + ': ' + str(v))

metrics = {'steps': [], 'rewards': [], 'Qs': [], 'best_avg_reward': -float('inf')}
np.random.seed(args.seed)
torch.manual_seed(np.random.randint(1, 10000))
args.device = torch.device('cpu')

# Agent
dqn_list = []
for _ in range(args.num_ensemble):
    dqn = Agent(env)
    dqn_list.append(dqn)
mem = ReplayMemory(args, args.memory_capacity, args.beta_mean, args.num_ensemble)

priority_weight_increase = (1 - args.priority_weight) / (args.T_max - args.learn_start)


rewards_test = []
rewards_test = np.zeros(EVALUATION_EPISODES)


# Construct validation memory
val_mem = ReplayMemory(args, args.evaluation_size, args.beta_mean, args.num_ensemble)
T, done = 0, True
while T < args.evaluation_size:
    if done:
        # state, done = env.reset(), False
        state_t, demand, source, destination = env.reset()
        state = get_state_feature(state_t, env, source, destination, demand)
    # next_state, _, done = env.step(np.random.randint(0, action_space))
    action = np.random.randint(0, ACTION_LEN)
    new_state_t, reward, done, demand, source, destination = env.make_step(state_t, action,demand,source,destination)
    next_state = get_state_feature(new_state_t, env, source, destination, demand)
    val_mem.append(state, None, None, done)
    state = next_state
    T += 1


# Training loop
T, done = 0, True
selected_en_index = np.random.randint(args.num_ensemble)

for T in trange(1, args.T_max + 1):
    if done:
        state, demand, source, destination = env.reset()
        done = False
        # state, done = env.reset(), False
        selected_en_index = np.random.randint(args.num_ensemble)

    if T % args.replay_frequency == 0:
        for en_index in range(args.num_ensemble):
            dqn_list[en_index].reset_noise()  # Draw a new set of noisy weights@

    # UCB exploration
    if args.ucb_infer > 0:
        mean_Q, var_Q = None, None
        L_target_Q = []
        for en_index in range(args.num_ensemble):
            target_Q, list_k_features = dqn_list[en_index].get_qvalues(env_eval, state, demand, source, destination, False)
            L_target_Q.append(target_Q)
            if en_index == 0:
                mean_Q = target_Q / args.num_ensemble
            else:
                mean_Q += target_Q / args.num_ensemble
        temp_count = 0
        for target_Q in L_target_Q:
            if temp_count == 0:
                var_Q = (target_Q - mean_Q) ** 2
            else:
                var_Q += (target_Q - mean_Q) ** 2
            temp_count += 1
        var_Q = var_Q / temp_count
        std_Q = torch.sqrt(var_Q).detach()
        ucb_score = mean_Q + args.ucb_infer * std_Q
        # 选择最大化UCB动作
        action = ucb_score.argmax(1)[0].item()
        state_action = list_k_features[action]
    else:
        # Choose an action greedily (with noisy weights)
        action, state_action = dqn_list[selected_en_index].act(env, state, demand, source, destination, False)

    # next_state, reward, done = env.step(action)  # Step
    new_state, reward, done, new_demand, new_source, new_destination = env.make_step(state, action, demand,
                                                                                              source, destination)
    if args.reward_clip > 0:
        reward = max(min(reward, args.reward_clip), -args.reward_clip)  # Clip rewards
    # mem.append(state, action, reward, done)  # Append transition to memory
    mem.append(state_action, action, reward, done, new_state, new_demand, new_source,new_destination)
    state = new_state
    demand = new_demand
    source = new_source
    destination = new_destination

    # Train and test
    if T >= args.learn_start:
        # Anneal importance sampling weight β to 1
        mem.priority_weight = min(mem.priority_weight + priority_weight_increase,1)
        # 回放

        if T % args.replay_frequency == 0:
            print("replay_frequency", args.replay_frequency)
            total_q_loss = 0

            # Sample transitions
            idxs, states, actions, returns, next_states, nonterminals, weights, masks = mem.sample(args.batch_size)
            q_loss_tot = 0

            weight_Q = None
            # Corrective feedback
            if args.temperature > 0:
                mean_Q, var_Q = None, None
                L_target_Q = []
                for en_index in range(args.num_ensemble):
                    target_Q = dqn_list[en_index].get_target_q(next_states)
                    L_target_Q.append(target_Q)
                    if en_index == 0:
                        mean_Q = target_Q / args.num_ensemble
                    else:
                        mean_Q += target_Q / args.num_ensemble
                temp_count = 0
                for target_Q in L_target_Q:
                    if temp_count == 0:
                        var_Q = (target_Q - mean_Q) ** 2
                    else:
                        var_Q += (target_Q - mean_Q) ** 2
                    temp_count += 1
                var_Q = var_Q / temp_count
                std_Q = torch.sqrt(var_Q).detach()
                weight_Q = torch.sigmoid(-std_Q * args.temperature) + 0.5

            for en_index in range(args.num_ensemble):
                # Train with n-step distributional double-Q learning
                q_loss = dqn_list[en_index].ensemble_learn(idxs, states, actions, returns,
                                                           next_states, nonterminals, weights,
                                                           masks[:, en_index], weight_Q)
                # grad, q_loss = dqn_list[en_index]._train_step(batch)
                if en_index == 0:
                    q_loss_tot = q_loss
                else:
                    q_loss_tot += q_loss
            q_loss_tot = q_loss_tot / args.num_ensemble

            # Update priorities of sampled transitions
            mem.update_priorities(idxs, q_loss_tot)

        if T % args.evaluation_interval == 0:
            print(T, "--评估--评估间隔", args.evaluation_interval)
            avg_reward, avg_Q = ensemble_eval(env_eval,T, dqn_list, val_mem, metrics,EVALUATION_EPISODES,
                                              num_ensemble=args.num_ensemble)  # Test
            # If memory path provided, save it
            if args.memory is not None:
                save_memory(mem, args.memory, args.disable_bzip_memory)

        # Update target network
        if T % args.target_update == 0:
            print("Update target network")
            for en_index in range(args.num_ensemble):
                dqn_list[en_index].update_target_net()
    state = next_state
env.close()
