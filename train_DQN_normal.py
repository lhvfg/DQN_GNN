from ES.DQNagent_normal import Agent as agent
from configs.dqn_normal import *
import gym
import tensorflow as tf
from util.utils import save_throughput
import numpy as np


hparams = {
    'link_state_dim': 20,
    'num_demands': len(listofDemands)
}


def get_graph_features(env, copyGraph):
    bw_allocated_feature = np.zeros((env_training.numEdges, len(env_training.listofDemands)))
    bw_allocated_feature.fill(0.0)
    # Normalize capacity feature
    capacity_feature = (copyGraph[:, 0] - 100.00000001) / 200.0

    iter = 0
    for i in copyGraph[:, 1]:
        if i == 8:
            bw_allocated_feature[iter][0] = 1
        elif i == 32:
            bw_allocated_feature[iter][1] = 1
        elif i == 64:
            bw_allocated_feature[iter][2] = 1
        iter = iter + 1

    sample = {
        'num_edges': env.numEdges,
        'length': env.firstTrueSize,
        'betweenness': tf.convert_to_tensor(value=env.between_feature, dtype=tf.float32),
        'bw_allocated': tf.convert_to_tensor(value=bw_allocated_feature, dtype=tf.float32),
        'capacities': tf.convert_to_tensor(value=capacity_feature, dtype=tf.float32),
        'first': tf.convert_to_tensor(env.first, dtype=tf.int32),
        'second': tf.convert_to_tensor(env.second, dtype=tf.int32)
    }

    sample['capacities'] = tf.reshape(sample['capacities'][0:sample['num_edges']], [sample['num_edges'], 1])
    sample['betweenness'] = tf.reshape(sample['betweenness'][0:sample['num_edges']], [sample['num_edges'], 1])

    hiddenStates = tf.concat([sample['capacities'], sample['betweenness'],sample['bw_allocated']], axis=1)

    paddings = tf.constant([[0, 0], [0, hparams['link_state_dim'] - 2 - hparams['num_demands']]])
    link_state = tf.pad(tensor=hiddenStates, paddings=paddings, mode="CONSTANT")
    return link_state


def get_state_feature(state,env,source,destination,demand):
    path = 0
    listGraphs = []
    # List of graph features that are used in the cummax() call
    list_k_features = list()
    pathList = env.allPaths[str(source) + ':' + str(destination)]
    while path < len(pathList):
        state_copy = np.copy(state)
        currentPath = pathList[path]
        i = 0
        j = 1

        # 3. Iterate over paths' pairs of nodes and allocate demand to bw_allocated
        while (j < len(currentPath)):
            state_copy[env.edgesDict[str(currentPath[i]) + ':' + str(currentPath[j])]][1] = demand
            i = i + 1
            j = j + 1
        # 4. Add allocated graphs' features to the list. Later we will compute their q-values using cummax
        listGraphs.append(state_copy)
        features = get_graph_features(env, state_copy)
        list_k_features.append(features)
        path = path + 1
    neg_numpy = np.array(list_k_features) # 4*21*20 action*边*特征
    list_k_features = torch.from_numpy(neg_numpy)    # action*边*特征 tensor(4,21,20)
    return list_k_features


def train(env_training, agent, num_episode, eps_init, eps_decay, eps_min, max_t):
    eps = eps_init
    for iter in range(num_episode):
        for _ in range(train_episodes):
            tf.random.set_seed(1)
            state, demand, source, destination = env_training.reset()
            while 1:
                # We execute evaluation over current state
                state_hidden = get_state_feature(state,env_training,source,destination,demand)
                action = agent.act(state_hidden, eps)
                new_state, reward, done, new_demand, new_source, new_destination = env_training.make_step(state, action,
                                                                                                          demand,
                                                                                                          source,
                                                                                                          destination)
                # 修改
                new_state_hidden = get_state_feature(new_state, env_training, source, destination, demand)
                agent.memory.append((state_hidden, action, reward, new_state_hidden,done))
                state = new_state
                demand = new_demand
                source = new_source
                destination = new_destination
                if done:
                    break
            # 到了应该学习的时间和足够案例
        if len(agent.memory) >= agent.bs:
            agent.learn()
            agent.soft_update(agent.tau)
        if iter == 600:
            dir = './models/model'+str(iter)+'-graph3.pth'
            torch.save(agent.Q_local, dir)
        demandlist = np.zeros(EVALUATION_EPISODES)
        min_utilization = np.zeros(EVALUATION_EPISODES)
        mean_utilization = np.zeros(EVALUATION_EPISODES)
        rewards_test = np.zeros(EVALUATION_EPISODES)
        for j in range(EVALUATION_EPISODES):
            state, demand, source, destination = env_eval.reset()
            rewardAddTest = 0
            alldemand = 0
            u1 = []
            min_u1 = []
            while 1:
                # We execute evaluation over current state
                state_hidden = get_state_feature(state, env_training, source, destination, demand)
                action = agent.act(state_hidden, eps)

                # 获取吞吐量
                alldemand += demand
                # 获取剩余带宽
                u = []
                min_u = []
                a = 0
                for i in (0, STATE_SIZE - 1):
                    a = (200 - state[i][0]) / 200
                    if (a > 1): a = 1.0
                    u.append(a)
                u = np.array(u)
                min_u.append(u.min())
                min_u = np.array(min_u)
                min_u = np.mean(min_u)
                u = np.mean(u)
                u1.append(u)
                min_u1.append(min_u)

                new_state, reward, done, demand, source, destination = env_eval.make_step(state, action, demand, source,
                                                                                          destination)
                rewardAddTest = rewardAddTest + reward
                state = new_state
                if done:
                    break
            u1 = np.array(u1)
            u1 = np.mean(u1)
            min_u1 = np.array(min_u1)
            min_u1 = np.mean(min_u1)
            demandlist[j] = alldemand
            mean_utilization[j] = u1
            min_utilization[j] = min_u1
            rewards_test[j] = rewardAddTest
        evalMeanReward = np.mean(rewards_test)
        save_throughput('DQN_DNN-' + str(graph_topology), eps, evalMeanReward, demandlist, min_utilization,
                      mean_utilization)

        eps = max(eps * eps_decay, eps_min)
        print("Training iteration", iter, ",reward:", evalMeanReward,",eps:",eps)


if __name__ == '__main__':

    env_training = gym.make(ENV_NAME)
    np.random.seed(SEED)
    env_training.seed(SEED)
    env_training.generate_environment(graph_topology, listofDemands)

    env_eval = gym.make(ENV_NAME)
    np.random.seed(SEED)
    env_eval.seed(SEED)
    env_eval.generate_environment(graph_topology, listofDemands)


    STATE_SIZE = env_training.numEdges
    ACTION_LEN = 4
    agent = agent(STATE_SIZE, ACTION_LEN, BATCH_SIZE, LEARNING_RATE, TAU, GAMMA, DEVICE,
                  False)
    train(env_training, agent, RAM_NUM_EPISODE, EPS_INIT, EPS_DECAY, EPS_MIN, MAX_T)
    agent.Q_local.to('cpu')
