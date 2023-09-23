import torch

ENV_NAME = 'GraphEnv-v1'
graph_topology = 0  # 0==NSFNET, 1==GEANT2, 2==Small Topology, 3==GBN
change_graph = 1
SEED = 37
listofDemands = [8, 32, 64]
train_episodes = 50
EVALUATION_EPISODES = 40

#device, CPU or the GPU of your choice
GPU = 0
DEVICE = device = torch.device("cpu")

#environment names
RAM_ENV_NAME = 'LunarLanderContinuous-v2'
VISUAL_ENV_NAME = None

#Agent parameters
BATCH_SIZE = 64
LR1 = 0.001
LR2 = 0.001
SPEED1 = 1
SPEED2 = 1
STEP = 1
TAU = 0.001
LEARNING_TIME = 1
OUN = True,
BN = False,
CLIP = False,
INIT = True,
HIDDEN = [256, 256]

#Training parameters
RAM_NUM_EPISODE = 2000  # 场景数
EPS_INIT = 1  # 随机动作概率初始值
EPS_DECAY = 0.995  # 随机动作概率消减值
EPS_MIN = 0.05  # 随机动作概率最小值
MAX_T = 100  # 每一场景下动作次数