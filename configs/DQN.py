
ENV_NAME = 'GraphEnv-v1'
graph_topology = 0 # 0==NSFNET, 1==GEANT2, 2==Small Topology, 3==GBN
SEED = 37
ITERATIONS = 2000
TRAINING_EPISODES = 50
EVALUATION_EPISODES = 40
FIRST_WORK_TRAIN_EPISODE = 60

MULTI_FACTOR_BATCH = 6 # Number of batches used in training
TAU = 0.08 # Only used in soft weights copy

differentiation_str = "my_DQN_agent"
checkpoint_dir = "./models"+differentiation_str
store_loss = 3 # Store the loss every store_loss batches

# Force TensorFlow to use single thread.
# Multiple threads are a potential source of non-reproducible results.
# For further details, see: https://stackoverflow.com/questions/42022950/
# tf.config.threading.set_inter_op_parallelism_threads(1)
# tf.config.threading.set_intra_op_parallelism_threads(1)

train_dir = "./TensorBoard/"+differentiation_str
# summary_writer = tf.summary.create_file_writer(train_dir)
listofDemands = [8, 32, 64]
copy_weights_interval = 50
evaluation_interval = 1
epsilon_start_decay = 70


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

MAX_QUEUE_SIZE = 4000