import numpy as np
import gc
import random
import DQN_MPNN.mpnn as gnn
import tensorflow as tf
from collections import deque
from DQN_MPNN.DQN_config import *


def cummax(alist, extractor):
    with tf.name_scope('cummax'):
        maxes = [tf.reduce_max(extractor(v)) + 1 for v in alist]
        cummaxes = [tf.zeros_like(maxes[0])]
        for i in range(len(maxes) - 1):
            cummaxes.append(tf.math.add_n(maxes[0:i + 1]))
    return cummaxes


class DQNAgent:
    def __init__(self, env):
        self.bw_allocated_feature = np.zeros((env.numEdges, len(listofDemands)))
        self.memory = deque(maxlen=MAX_QUEUE_SIZE)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        #  0.995
        self.epsilon_decay = 0.998
        self.writer = None
        self.K = 4  # K-paths
        self.listQValues = None
        self.numbersamples = hparams['batch_size']
        self.action = None
        self.capacity_feature = None

        self.global_step = 0
        self.primary_network = gnn.myModel(hparams)
        self.primary_network.build()
        self.target_network = gnn.myModel(hparams)
        self.target_network.build()
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=hparams['learning_rate'], momentum=0.9, nesterov=True)

    def get_qvalues(self, env, state, demand, source, destination, flagEvaluation):
        """
        给定存储在环境中的需求，它在当前“状态”上分配K=4条最短路径，并利用GNN模型预测了K=4种不同新图形状态的q_值。
        根据epsilon贪婪方法选择状态。标志=TRUE表示我们正在测试，因此，它不会激活放置层。
        """
        # Set to True if we need to compute K=4 q-values and take the maxium
        takeMax_epsilon = False
        # List of graphs
        listGraphs = []
        # List of graph features that are used in the cummax() call
        list_k_features = list()
        # We get the K-paths between source-destination
        pathList = env.allPaths[str(source) + ':' + str(destination)]
        path = 0

        # 1. Implement epsilon-greedy to pick allocation
        # If flagEvaluation==TRUE we are EVALUATING => take always the action that the agent is saying has higher q-value
        # Otherwise, we are training with normal epsilon-greedy strategy
        if flagEvaluation:
            # If evaluation, compute K=4 q-values and take the maxium value
            takeMax_epsilon = True
        else:
            # If training, compute epsilon-greedy
            z = np.random.random()
            if z > self.epsilon:
                # Compute K=4 q-values and pick the one with highest value
                # In case of multiple same max values, return the first one
                takeMax_epsilon = True
            else:
                # Pick a random path and compute only one q-value
                path = np.random.randint(0, len(pathList))

        # 2. Allocate (S,D, linkDemand) demand using the K shortest paths
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
            features = self.get_graph_features(env, state_copy)
            list_k_features.append(features)

            if not takeMax_epsilon:
                # If we don't need to compute the K=4 q-values we exit
                break

            path = path + 1

        vs = [v for v in list_k_features]

        # We compute the graphs_ids to later perform the unsorted_segment_sum for each graph and obtain the
        # link hidden states for each graph.
        graph_ids = [tf.fill([tf.shape(vs[it]['link_state'])[0]], it) for it in range(len(list_k_features))]
        first_offset = cummax(vs, lambda v: v['first'])
        second_offset = cummax(vs, lambda v: v['second'])

        tensors = ({
            'graph_id': tf.concat([v for v in graph_ids], axis=0),
            'link_state': tf.concat([v['link_state'] for v in vs], axis=0),
            'first': tf.concat([v['first'] + m for v, m in zip(vs, first_offset)], axis=0),
            'second': tf.concat([v['second'] + m for v, m in zip(vs, second_offset)], axis=0),
            'num_edges': tf.math.add_n([v['num_edges'] for v in vs]),
        }
        )

        # Predict qvalues for all graphs within tensors
        self.listQValues = self.primary_network(tensors['link_state'], tensors['graph_id'], tensors['first'],
                                                tensors['second'], tensors['num_edges'], training=False).numpy()
        return self.listQValues

    def act(self, env, state, demand, source, destination, flagEvaluation):
        """
        给定存储在环境中的需求，它在当前“状态”上分配K=4条最短路径，并利用GNN模型预测了K=4种不同新图形状态的q_值。
        根据epsilon贪婪方法选择状态。标志=TRUE表示我们正在测试，因此，它不会激活放置层。
        """
        # Set to True if we need to compute K=4 q-values and take the maxium
        takeMax_epsilon = False
        # List of graphs
        listGraphs = []
        # List of graph features that are used in the cummax() call
        list_k_features = list()
        # Initialize action
        action = 0

        # We get the K-paths between source-destination
        pathList = env.allPaths[str(source) + ':' + str(destination)]
        path = 0

        # 1. Implement epsilon-greedy to pick allocation
        # If flagEvaluation==TRUE we are EVALUATING => take always the action that the agent is saying has higher q-value
        # Otherwise, we are training with normal epsilon-greedy strategy
        if flagEvaluation:
            # If evaluation, compute K=4 q-values and take the maxium value
            takeMax_epsilon = True
        else:
            # If training, compute epsilon-greedy
            z = np.random.random()
            if z > self.epsilon:
                # Compute K=4 q-values and pick the one with highest value
                # In case of multiple same max values, return the first one
                takeMax_epsilon = True
            else:
                # Pick a random path and compute only one q-value
                path = np.random.randint(0, len(pathList))
                action = path

        # 2. Allocate (S,D, linkDemand) demand using the K shortest paths
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
            features = self.get_graph_features(env, state_copy)
            list_k_features.append(features)

            if not takeMax_epsilon:
                # If we don't need to compute the K=4 q-values we exit
                break

            path = path + 1

        vs = [v for v in list_k_features]

        # We compute the graphs_ids to later perform the unsorted_segment_sum for each graph and obtain the
        # link hidden states for each graph.
        graph_ids = [tf.fill([tf.shape(vs[it]['link_state'])[0]], it) for it in range(len(list_k_features))]
        first_offset = cummax(vs, lambda v: v['first'])
        second_offset = cummax(vs, lambda v: v['second'])

        tensors = ({
            'graph_id': tf.concat([v for v in graph_ids], axis=0),
            'link_state': tf.concat([v['link_state'] for v in vs], axis=0),
            'first': tf.concat([v['first'] + m for v, m in zip(vs, first_offset)], axis=0),
            'second': tf.concat([v['second'] + m for v, m in zip(vs, second_offset)], axis=0),
            'num_edges': tf.math.add_n([v['num_edges'] for v in vs]),
        }
        )

        # Predict qvalues for all graphs within tensors
        self.listQValues = self.primary_network(tensors['link_state'], tensors['graph_id'], tensors['first'],
                                                tensors['second'], tensors['num_edges'], training=False).numpy()

        if takeMax_epsilon:
            # We take the path with highest q-value
            action = np.argmax(self.listQValues)
        else:
            return path, list_k_features[0]
        return action, list_k_features[action]

    def get_graph_features(self, env, copyGraph):
        self.bw_allocated_feature.fill(0.0)
        # Normalize capacity feature
        self.capacity_feature = (copyGraph[:, 0] - 100.00000001) / 200.0

        iter = 0
        for i in copyGraph[:, 1]:
            if i == 8:
                self.bw_allocated_feature[iter][0] = 1
            elif i == 32:
                self.bw_allocated_feature[iter][1] = 1
            elif i == 64:
                self.bw_allocated_feature[iter][2] = 1
            iter = iter + 1

        sample = {
            'num_edges': env.numEdges,
            'length': env.firstTrueSize,
            'betweenness': tf.convert_to_tensor(value=env.between_feature, dtype=tf.float32),
            'bw_allocated': tf.convert_to_tensor(value=self.bw_allocated_feature, dtype=tf.float32),
            'capacities': tf.convert_to_tensor(value=self.capacity_feature, dtype=tf.float32),
            'first': tf.convert_to_tensor(env.first, dtype=tf.int32),
            'second': tf.convert_to_tensor(env.second, dtype=tf.int32)
        }

        sample['capacities'] = tf.reshape(sample['capacities'][0:sample['num_edges']], [sample['num_edges'], 1])
        sample['betweenness'] = tf.reshape(sample['betweenness'][0:sample['num_edges']], [sample['num_edges'], 1])

        hiddenStates = tf.concat([sample['capacities'], sample['betweenness'], sample['bw_allocated']], axis=1)

        paddings = tf.constant([[0, 0], [0, hparams['link_state_dim'] - 2 - hparams['num_demands']]])
        link_state = tf.pad(tensor=hiddenStates, paddings=paddings, mode="CONSTANT")

        inputs = {'link_state': link_state, 'first': sample['first'][0:sample['length']],
                  'second': sample['second'][0:sample['length']], 'num_edges': sample['num_edges']}

        return inputs

    @tf.function
    def _forward_pass(self, x):
        prediction_state = self.primary_network(x[0], x[1], x[2], x[3], x[4], training=True)
        preds_next_target = tf.stop_gradient(self.target_network(x[6], x[7], x[9], x[10], x[11], training=True))
        return prediction_state, preds_next_target

    def _train_step(self, batch):
        # Record operations for automatic differentiation
        with tf.GradientTape() as tape:
            preds_state = []
            target = []
            for x in batch:
                prediction_state, preds_next_target = self._forward_pass(x)
                # Take q-value of the action performed
                preds_state.append(prediction_state[0])
                # We multiple by 0 if done==TRUE to cancel the second term
                target.append(
                    tf.stop_gradient([x[5] + self.gamma * tf.math.reduce_max(preds_next_target) * (1 - x[8])]))

            loss = tf.keras.losses.MSE(tf.stack(target, axis=1), tf.stack(preds_state, axis=1))
            # Loss function using L2 Regularization
            regularization_loss = sum(self.primary_network.losses)
            loss = loss + regularization_loss

        # Computes the gradient using operations recorded in context of this tape
        grad = tape.gradient(loss, self.primary_network.variables)
        # gradients, _ = tf.clip_by_global_norm(grad, 5.0)
        gradients = [tf.clip_by_value(gradient, -1., 1.) for gradient in grad]
        self.optimizer.apply_gradients(zip(gradients, self.primary_network.variables))
        del tape
        return grad, loss

    def replay(self, episode):
        for i in range(MULTI_FACTOR_BATCH):
            batch = random.sample(self.memory, self.numbersamples)

            grad, loss = self._train_step(batch)
            # if i % store_loss == 0:
            #     fileLogs.write(".," + '%.9f' % loss.numpy() + ",\n")

        # Soft weights update
        # for t, e in zip(self.target_network.trainable_variables, self.primary_network.trainable_variables):
        #     t.assign(t * (1 - TAU) + e * TAU)

        # Hard weights update
        if episode % copy_weights_interval == 0:
            self.target_network.set_weights(self.primary_network.get_weights())
            # if episode % evaluation_interval == 0:
        gc.collect()

    def add_sample(self, env_training, state_action, action, reward, done, new_state, new_demand, new_source,
                   new_destination):
        self.bw_allocated_feature.fill(0.0)
        new_state_copy = np.copy(new_state)

        state_action['graph_id'] = tf.fill([tf.shape(state_action['link_state'])[0]], 0)

        # We get the K-paths between new_source-new_destination
        pathList = env_training.allPaths[str(new_source) + ':' + str(new_destination)]
        path = 0
        list_k_features = list()

        # 2. Allocate (S,D, linkDemand) demand using the K shortest paths
        while path < len(pathList):
            currentPath = pathList[path]
            i = 0
            j = 1

            # 3. Iterate over paths' pairs of nodes and allocate new_demand to bw_allocated
            while (j < len(currentPath)):
                new_state_copy[env_training.edgesDict[str(currentPath[i]) + ':' + str(currentPath[j])]][1] = new_demand
                i = i + 1
                j = j + 1

            # 4. Add allocated graphs' features to the list. Later we will compute it's qvalues using cummax
            features = self.get_graph_features(env_training, new_state_copy)

            list_k_features.append(features)
            path = path + 1
            new_state_copy[:, 1] = 0

        vs = [v for v in list_k_features]

        # We compute the graphs_ids to later perform the unsorted_segment_sum for each graph and obtain the
        # link hidden states for each graph.
        graph_ids = [tf.fill([tf.shape(vs[it]['link_state'])[0]], it) for it in range(len(list_k_features))]
        first_offset = cummax(vs, lambda v: v['first'])
        second_offset = cummax(vs, lambda v: v['second'])

        tensors = ({
            'graph_id': tf.concat([v for v in graph_ids], axis=0),
            'link_state': tf.concat([v['link_state'] for v in vs], axis=0),
            'first': tf.concat([v['first'] + m for v, m in zip(vs, first_offset)], axis=0),
            'second': tf.concat([v['second'] + m for v, m in zip(vs, second_offset)], axis=0),
            'num_edges': tf.math.add_n([v['num_edges'] for v in vs]),
        }
        )

        # We store the state with the action marked, the graph ids, first, second, num_edges, the reward,
        # new_state(-1 because we don't need it in this case), the graph ids, done, first, second, number of edges
        self.memory.append((state_action['link_state'], state_action['graph_id'], state_action['first'],  # 2
                            state_action['second'], tf.convert_to_tensor(state_action['num_edges']),  # 4
                            tf.convert_to_tensor(reward, dtype=tf.float32), tensors['link_state'], tensors['graph_id'],
                            # 7
                            tf.convert_to_tensor(int(done == True), dtype=tf.float32), tensors['first'],
                            tensors['second'],  # 10
                            tf.convert_to_tensor(tensors['num_edges'])))  # 12
