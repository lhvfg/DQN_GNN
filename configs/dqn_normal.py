import torch

ENV_NAME = 'GraphEnv-v1'
graph_topology = 3  # 0==NSFNET, 1==GEANT2, 2==Small Topology, 3==GBN
SEED = 37
listofDemands = [8, 32, 64]
train_episodes = 50
EVALUATION_EPISODES = 40

#device, CPU or the GPU of your choice
GPU = 0
DEVICE = torch.device("cuda:{}".format(GPU) if torch.cuda.is_available() else "cpu")

#env names
MY_ENV = 'PPO_ENV'
CONSTANT = 90

#Agent parameters
STATE_SIZE = 210
ACTION_SIZE = 10
ACTION_LEN = 42  # 连续动作个数即链路数

BATCH_SIZE = 128  # 经验池

LEARNING_RATE = 0.0001  # 学习率
TAU = 0.001  # 线性计算q_local
GAMMA = 0.9  # 更新q_target

#Training parameters
RAM_NUM_EPISODE = 2000  # 场景数
EPS_INIT = 1  # 随机动作概率初始值
EPS_DECAY = 0.998  # 随机动作概率消减值
EPS_MIN = 0.05  # 随机动作概率最小值
MAX_T = 100  # 每一场景下动作次数
NUM_FRAME = 2  # 实际可能做了2-6个frame
