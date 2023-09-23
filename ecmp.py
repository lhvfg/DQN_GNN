import random

import gym
import numpy as np
from util.utils import save,save_throughput
import tensorflow as tf

ENV_NAME = 'GraphEnv-v1'   
graph_topology = 0 # 0==NSFNET, 1==GEANT2, 2==Small Topology, 3==GBN
SEED = 37
listofDemands = [8, 32, 64]
ITERATIONS = 2000
EVALUATION_EPISODES = 50
#__name__ 是一个在 Python 中预定义的特殊变量，用来表示当前模块的名称当一个模块被直接运行时（即作为主程序运行），__name__ 的值会被自动设置为字符串 "__main__"。
# 当一个模块被其他模块导入时，__name__ 的值会被设置为模块的名称。
# 通过使用 if __name__ == "__main__" 的条件判断，我们可以确定当前模块是作为主程序运行还是作为模块被导入，并相应地执行不同的代码块。
# 例如，当我们直接运行一个 Python 脚本时，__name__ 的值将是 "__main__"，因此 if __name__ == "__main__" 条件将会成立，对应的代码块将会被执行。但是，如果这个脚本被其他模块导入使用，则 __name__ 的值将是该模块的名称，该条件判断将为假，对应的代码块将不会被执行。
# 使用 __name__ 变量可以方便地控制模块的运行行为，并确保模块在导入时不会执行不需要的代码。
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
                # 获取吞吐量
                alldemand += demand
                new_state, reward, done, new_demand, new_source, new_destination = env_training.ecmp_step(state, demand, source, destination)
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
        print("Training iteration", ep_it, "reward:", rewardAddTest)
        save_throughput('ecmp'+str(graph_topology), ep_it, evalMeanReward,demandlist, min_utilization,
                          mean_utilization)
