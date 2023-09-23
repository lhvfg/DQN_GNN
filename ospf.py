import random

import gym
import numpy as np
from util.utils import save,save_throughput
import tensorflow as tf

ENV_NAME = 'GraphEnv-v1'
graph_topology = 3 # 0==NSFNET, 1==GEANT2, 2==Small Topology, 3==GBN
SEED = 37
listofDemands = [8, 32, 64]
ITERATIONS = 2000
EVALUATION_EPISODES = 50

def select_action():
    pathList = env_training.allPaths[str(source) + ':' + str(destination)]
    # 随机选取action试试
    distances = []
    for path in pathList:
        # distance = 0
        # for node in range(len(path) - 1):
        #     distance += env_training.edgesDict[str(path[node]) + ':' + str(path[node + 1])]
        # distances.append(distance)
        distances.append(len(path))
    return np.argmin(distances)

if __name__ == "__main__":
    env_training = gym.make(ENV_NAME)
    env_training.seed(SEED)
    env_training.generate_environment(graph_topology, listofDemands)
    state_size = env_training.numEdges
    rewards_test = []
    rewards_test = np.zeros(EVALUATION_EPISODES)
    for ep_it in range(ITERATIONS):
        # Used to clean the TF cache
        demandlist = np.zeros(EVALUATION_EPISODES)
        min_utilization = np.zeros(EVALUATION_EPISODES)
        mean_utilization = np.zeros(EVALUATION_EPISODES)
        rewards_test = np.zeros(EVALUATION_EPISODES)
        tf.random.set_seed(1)
        for eps in range(EVALUATION_EPISODES):
            state, demand, source, destination = env_training.reset()
            traffic_sum = demand
            rewardAddTest = 0
            alldemand = 0
            while 1:
                action = select_action()

                # 获取吞吐量
                alldemand += demand

                new_state, reward, done, new_demand, new_source, new_destination = env_training.make_step(state, action, demand, source, destination)
                rewardAddTest += reward
                traffic_sum += new_demand

                state = new_state
                demand = new_demand
                source = new_source
                destination = new_destination
                if done:
                    break
            # 获取剩余带宽
            u = []
            min_u = np.zeros(1)
            a = 0
            for i in range(state_size):
                a = (200 - state[i][0]) / 200
                if (a > 1): a = 1.0
                u.append(a)

            np.array(u)
            min_u = np.min(u)
            u = np.mean(u)

            demandlist[eps] = alldemand
            mean_utilization[eps] = u
            min_utilization[eps] = min_u
            rewards_test[eps] = rewardAddTest
        evalMeanReward = np.mean(rewards_test)
        print("Training iteration", ep_it, "reward:", evalMeanReward)
        save_throughput('ospf-'+str(graph_topology), ep_it, evalMeanReward,demandlist, min_utilization,
                          mean_utilization)
