import numpy as np
from collections import namedtuple
from numpy import array, float32

from src.PID import discretize

def get_FF_activations(df):
    """
    """
    transition = namedtuple('transition', ('observation', 'activation', 'reward'))
    configuration = namedtuple('configuration', ('episode', 'buffer', 'success'))
    model_info = namedtuple('model_info', ('configuration_file', 'configuration_buffer'))

    list_df = df.values.tolist()
    activations = {}
    inputs = []
    layers = []
    outputs = []
    total_reward = 0
    num_success = 0
    total_steps = 0
    total_episodes = 0

    # each configuration
    for i in range(len(list_df)): 
        configuration_tuple = eval(list_df[i][2])
        # each episode in configuration
        for j in range(len(configuration_tuple)):
            buffer = configuration_tuple[j].buffer
            total_episodes += 1
            if configuration_tuple[j].success:
                num_success += 1
            if i == 0:
                for t in range(len(buffer[0].activation)-1):
                    layers.append([])

            # each environment transition step in episode
            for k in range(len(buffer)):
                # network input
                observation = buffer[k].observation
                # activations
                activation = buffer[k].activation
                total_reward += buffer[k].reward
                total_steps += 1

                inputs.append(observation.flatten())

                for l in range(0, len(layers)):
                    layers[l].append(activation[l].flatten())
                outputs.append(np.argmax(activation[-1]).flatten())
    
    activations = {
        'inputs' : inputs,
        'layers' : layers,
        'outputs' : outputs,
        ## use avg_reward as accuracy for logic gate experiments
        'avg_reward' : total_reward/total_steps, # NOT avg episode reward- avg step reward
        'accuracy' : num_success/total_episodes
    }

    return activations


def get_RNN_activations(df): 
    """
    """
    transition = namedtuple('transition', ('observation', 'activation', 'h_activation', 'o_activation', 'reward'))
    configuration = namedtuple('configuration', ('episode', 'buffer', 'success'))
    model_info = namedtuple('model_info', ('configuration_file', 'configuration_buffer'))

    list_df = df.values.tolist()
    activations = {}
    inputs = []
    hidden_t = []
    hidden_tp1 = []
    ff_layer = []
    outputs = []
    trajectory = []
    cc_trajectory = []
    total_reward = 0
    num_success = 0
    total_steps = 0
    total_episodes = 0

    # each configuration
    max_buffer_len = 0
    for i in range(len(list_df)): 
        configuration_tuple = eval(list_df[i][2])
        # each episode in configuration
        buffer_len = 0
        for j in range(len(configuration_tuple)):
            buffer = configuration_tuple[j].buffer
            total_episodes += 1
            if configuration_tuple[j].success:
                num_success += 1
            if len(buffer) > buffer_len: 
                for t in range(4*(len(buffer)-buffer_len)):
                    trajectory.append([])
                buffer_len = len(buffer)
            if len(buffer) > max_buffer_len: 
                for t in range(4*(len(buffer)-max_buffer_len)):
                    cc_trajectory.append([])
                max_buffer_len = len(buffer)
            # each environment transition step in episode
            for k in range(len(buffer)):
                # network input
                observation = buffer[k].observation
                # activations
                activation = buffer[k].activation 
                h_activation = buffer[k].h_activation 
                o_activation = buffer[k].o_activation 
                total_reward += buffer[k].reward
                total_steps += 1

                if k == 0:
                    h_prev = np.zeros(h_activation.flatten().shape)
                hidden_t.append(h_prev)
                hidden_tp1.append(h_activation.flatten())
                
                inputs.append(observation.flatten())
                ff_layer.append(activation.flatten())
                outputs.append(o_activation.flatten())

                trajectory[k*4].append(observation.flatten())
                trajectory[k*4 + 1].append(h_prev)
                trajectory[k*4 + 2].append(o_activation.flatten()) 
                trajectory[k*4 + 3].append(h_activation.flatten())
                cc_trajectory[k*4].append(observation.flatten())
                cc_trajectory[k*4 + 1].append(h_prev)
                cc_trajectory[k*4 + 2].append(o_activation.flatten()) 
                cc_trajectory[k*4 + 3].append(h_activation.flatten())

                h_prev = h_activation.flatten()
        trajectory = []

    activations['inputs'] = inputs
    activations['hidden_t'] = hidden_t
    activations['hidden_tp1'] = hidden_tp1
    activations['ff_layer'] = ff_layer 
    activations['outputs'] = outputs
    activations['avg_reward'] = total_reward/total_steps
    activations['accuracy'] = num_success/total_episodes
    activations['cc_trajectory'] = cc_trajectory

    return activations


def get_activations(model_df, args):
    if args.network_type == 'feedforward':
        activations = get_FF_activations(model_df)
    elif args.network_type == 'recurrent':
        activations = get_RNN_activations(model_df)
    
    if args.discrete_PID:
        activations = discretize.discretize_activations(activations, args)

    return activations