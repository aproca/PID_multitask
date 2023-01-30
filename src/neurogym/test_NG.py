import numpy as np
import pandas as pd
import torch
from collections import namedtuple

from src.neurogym import NG_utils 
from src import utils

def test_config(env, model, args, indicator):
    # tests a single configuration and extracts activations
    transition = namedtuple('transition', ('observation', 'activation', 'h_activation', 'o_activation', 'reward'))
    configuration = namedtuple('configuration', ('episode', 'buffer', 'success'))

    config_buffer, episode_buffer = [], []
    ep_solved = 0
    _ = env.reset()
    model.eval()
    with torch.no_grad():
        for episode in range(args.ep_per_config_test):
            env.new_trial()
            observation, labels = env.ob, env.gt
            observation = observation[:, np.newaxis, :]
            observation = np.insert(observation, len(observation[0][0]), indicator, axis=2)
            inputs = torch.from_numpy(observation).type(torch.float)
            predictions = np.argmax(model(inputs).detach().numpy(), axis=-1)
            correct = labels[-1] == predictions[-1, 0]
            ep_solved += correct
            for t in range(inputs.shape[0]):
                reward = int(predictions[t] == labels[t])
                episode_buffer.append(transition(observation[t], model.activations[t], model.h_activations[t], model.o_activations[t], reward))
            
            config_buffer.append(configuration(episode, episode_buffer, bool(correct)))
            episode_buffer = []
    
    print('Configuration Accuracy {}'.format(ep_solved/args.ep_per_config_test))
    return config_buffer, ep_solved

def test(args):
    model_info = namedtuple('model_info', ('configuration_file', 'configuration_buffer'))
    utils.set_seeds(args.seed)

    model = torch.load(args.model_out_dir + args.base_folder + args.base_file)
    model.batch_size = 1
    
    curr = args.curriculum
    print('Testing on {}'.format(curr['name']))
    model_buffer = []
    tot_ep_solved = 0

    for config in range(len(curr['config_files'])):
        print('Configuration {}'.format(config))
        config_file = curr['config_files'][config]
        dataset = NG_utils.load_dataset(config_file, args)
        config_buffer, ep_solved = test_config(dataset.env, model, args, curr['indicators'][config])
        tot_ep_solved += ep_solved
        model_buffer.append(model_info(config_file, config_buffer))
        dataset.env.close()

    print('Test Accuracy {}'.format(tot_ep_solved/(len(curr['config_files'])*args.ep_per_config_test)))

    df = pd.DataFrame.from_records(model_buffer)
    PATH = utils.check_path(args.activation_out_dir + args.base_folder + args.ckpt_folder)
    df.to_csv(PATH + args.base_file + '.csv')
    

