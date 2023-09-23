import torch

#device, CPU or the GPU of your choice
GPU = 0
DEVICE = torch.device("cuda:{}".format(GPU) if torch.cuda.is_available() else "cpu")

ENV_NAME = 'GraphEnv-v1'
graph_topology = 1  # 0==NSFNET, 1==GEANT2, 2==Small Topology, 3==GBN
SEED = 37
listofDemands = [8, 32, 64]
train_episodes = 50
EVALUATION_EPISODES = 40
#environment names
CONSTANT = 90

#Agent parameters
LEARNING_RATE = 0.005
GAMMA = 0.99
BETA = 0
EPS = 0.2
TAU = 0.99
MODE = 'TD'
SHARE = False
CRITIC = True
NORMALIZE = False
HIDDEN_DISCRETE = [256,256]
BATCH_SIZE = 64

#Training parameters
RAM_NUM_EPISODE = 2000
SCALE = 1
MAX_T = 1000
NUM_FRAME = 2
N_UPDATE = 10
UPDATE_FREQUENCY = 4

hparams = {
    'l2': 0.1,
    'dropout_rate': 0.01,
    'link_state_dim': 20,
    'readout_units': 35,
    'learning_rate': 0.0001,
    'batch_size': 32,
    'T': 4,
    'num_demands': len(listofDemands)
}