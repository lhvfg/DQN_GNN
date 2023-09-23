
ENV_NAME = 'GraphEnv-v1'
graph_topology = 3 # 0==NSFNET, 1==GEANT2, 2==Small Topology, 3==GBN
SEED = 37
ITERATIONS = 2000
TRAINING_EPISODES = 50
EVALUATION_EPISODES = 40

MULTI_FACTOR_BATCH = 6 # Number of batches used in training
TAU = 0.08 # Only used in soft weights copy

store_loss = 3 # Store the loss every store_loss batches


listofDemands = [8, 32, 64]
copy_weights_interval = 50
evaluation_interval = 1
epsilon_start_decay = 70
num_ensemble = 2
ucb_infer = 1 # 1
# 备份的样本权重参数
temperature = 0.05
beta_mean = 1
hparams = {
    'l2': 0.1,
    'dropout_rate': 0.01,
    'link_state_dim': 20,
    'readout_units': 35,
    'learning_rate': 0.001,
    'batch_size': 32,
    'T': 4,
    'num_demands': len(listofDemands),
}

MAX_QUEUE_SIZE = 4000