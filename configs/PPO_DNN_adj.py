import torch

#device, CPU or the GPU of your choice
GPU = 0
DEVICE = torch.device("cuda:{}".format(GPU) if torch.cuda.is_available() else "cpu")

ENV_NAME = 'GraphEnv-v1'
graph_topology = 3  # 0==NSFNET, 1==GEANT2, 2==Small Topology, 3==GBN
SEED = 37
listofDemands = [8, 32, 64]
train_episodes = 50
EVALUATION_EPISODES = 40

EPS = 0.4
HIDDEN_DISCRETE = [256,256]
#Agent parameters
LEARNING_RATE = 0.1
GAMMA = 0.99 # 折扣因子
BETA = 0

TAU = 0.99
MODE = 'TD'
SHARE = False
CRITIC = True
NORMALIZE = False
BATCH_SIZE = 128

#Training parameters
RAM_NUM_EPISODE = 2000
SCALE = 1
N_UPDATE = 20

hparams = {
    'link_state_dim': 20,
    'num_demands': len(listofDemands)
}