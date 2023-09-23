# Copyright (c) 2021, Paul Almasan [^1]
#
# [^1]: Universitat Politècnica de Catalunya, Computer Architecture
#     department, Barcelona, Spain. Email: felician.paul.almasan@upc.edu

import tensorflow as tf
from tensorflow import keras
from keras import regularizers
import numpy as np

class myModel(tf.keras.Model):
    def __init__(self, hparams):
        super(myModel, self).__init__()
        self.hparams = hparams

        # Define layers here
        self.Message = tf.keras.models.Sequential()
        self.Message.add(keras.layers.Dense(self.hparams['link_state_dim'],
                                            activation=tf.nn.selu, name="FirstLayer"))
        self.Update = tf.keras.layers.GRUCell(self.hparams['link_state_dim'], dtype=tf.float32)

        self.Readout = tf.keras.models.Sequential()
        self.Readout.add(keras.layers.Dense(self.hparams['readout_units'],
                                            activation=tf.nn.selu,
                                            kernel_regularizer=regularizers.l2(hparams['l2']),
                                            name="Readout1"))
        self.Readout.add(keras.layers.Dropout(rate=hparams['dropout_rate']))
        self.Readout.add(keras.layers.Dense(self.hparams['readout_units'],
                                            activation=tf.nn.selu,
                                            kernel_regularizer=regularizers.l2(hparams['l2']),
                                            name="Readout2"))
        self.Readout.add(keras.layers.Dropout(rate=hparams['dropout_rate']))
        self.Readout.add(keras.layers.Dense(1, kernel_regularizer=regularizers.l2(hparams['l2']),
                                            name="Readout3"))

    def build(self, input_shape=None):
        self.Message.build(input_shape=tf.TensorShape([None, self.hparams['link_state_dim']*2]))
        self.Update.build(input_shape=tf.TensorShape([None,self.hparams['link_state_dim']]))
        self.Readout.build(input_shape=[None, self.hparams['link_state_dim']])
        self.built = True

    @tf.function
    def call(self, states_action, states_graph_ids, states_first, states_second, sates_num_edges, training=False):
        # Define the forward pass
        link_state = states_action

        # Execute T times
        for _ in range(self.hparams['T']):
            # We have the combination of the hidden states of the main edges with the neighbours
            mainEdges = tf.gather(link_state, states_first)
            neighEdges = tf.gather(link_state, states_second)

            edgesConcat = tf.concat([mainEdges, neighEdges], axis=1)

            ### 1.a Message passing for link with all it's neighbours
            outputs = self.Message(edgesConcat)

            ### 1.b Sum of output values according to link id index
            edges_inputs = tf.math.unsorted_segment_sum(data=outputs, segment_ids=states_second,
                                                        num_segments=sates_num_edges)

            ### 2. Update for each link
            # GRUcell needs a 3D tensor as state because there is a matmul: Wrap the link state
            outputs, links_state_list = self.Update(edges_inputs, [link_state])

            link_state = links_state_list[0]

        # Perform sum of all hidden states
        edges_combi_outputs = tf.math.segment_sum(link_state, states_graph_ids, name=None)

        r = self.Readout(edges_combi_outputs,training=training)
        return r

    @staticmethod
    def _get_specific_number_weights(model):
        weights = model.get_weights()
        layer_dimensions = [(w.shape, w.size) for w in weights]
        return layer_dimensions, sum(w[1] for w in layer_dimensions)

    def get_message_number_weights(self):
        return self._get_specific_number_weights(self.Message)

    def get_update_number_weights(self):
        return self._get_specific_number_weights(self.Update)

    def get_message_update_number_weights(self):
        message_layer_dimensions, message_number_params = self._get_specific_number_weights(self.Message)
        update_layer_dimensions, update_number_params = self._get_specific_number_weights(self.Update)
        return message_layer_dimensions + update_layer_dimensions, message_number_params + update_number_params

    def get_readout_number_weights(self):
        return self._get_specific_number_weights(self.Readout)

    def get_number_weights(self):
        return self._get_specific_number_weights(super(myModel, self))

    @staticmethod
    def _get_specific_weights(model):
        weights = model.get_weights()
        for w in range(len(weights)):
            weights[w] = np.reshape(weights[w], (weights[w].size,))
        return np.concatenate(weights)

    def get_message_weights(self):
        return self._get_specific_weights(self.Message)

    def get_update_weights(self):
        return self._get_specific_weights(self.Update)

    def get_message_update_weights(self):
        return np.concatenate((self._get_specific_weights(self.Message), self._get_specific_weights(self.Update)))

    def get_readout_weights(self):
        return self._get_specific_weights(self.Readout)

    def get_weights(self):
        return self._get_specific_weights(super(myModel, self))

    @staticmethod
    def _set_weights(model, new_weights):
        weights = model.get_weights()
        layer_dimensions = [(w.shape, w.size) for w in weights]

        transformed_weights = []
        current_idx = 0
        for layer_shape, layer_size in layer_dimensions:
            layer_weights = np.reshape(new_weights[current_idx:current_idx + layer_size], layer_shape)
            transformed_weights.append(layer_weights)
            current_idx += layer_size

        model.set_weights(transformed_weights)

    def set_message_weights(self, new_weights):
        self._set_weights(self.Message, new_weights)

    def set_update_weights(self, new_weights):
        self._set_weights(self.Update, new_weights)

    def set_message_update_weights(self, new_weights):
        _, message_number_params = self.get_message_number_weights()
        self._set_weights(self.Message, new_weights[:message_number_params])
        self._set_weights(self.Update, new_weights[message_number_params:])

    def set_readout_weights(self, new_weights):
        self._set_weights(self.Readout, new_weights)

    def set_weights(self, new_weights):
        self._set_weights(super(myModel, self), new_weights)

    def reset_noise(self):
        for name, module in self.named_children():
            if 'Readout' in name:
                module.reset_noise()