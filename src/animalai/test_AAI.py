import pandas as pd
from collections import namedtuple
from stable_baselines3 import PPO
from animalai.envs.arena_config import ArenaConfig

from src import utils
from src.animalai import AAI_utils

def test_config(env, model, args):
    # Tests a single configuration and extracts activations
    transition = namedtuple('transition', ('observation', 'activation', 'reward'))
    configuration = namedtuple('configuration', ('episode', 'buffer', 'success'))

    config_buffer, episode_buffer = [], []
    ep_solved = 0
    done = False
    observation = env.reset()
    for episode in range(args.ep_per_config_test):
        while not done:
            action, _ = model.predict(observation, deterministic=True)
            activations = model.policy.mlp_extractor.activations
            observation, reward, done, _ = env.step(action)
            if done:
                if reward > -0.0007:
                    success = True
                    ep_solved += 1
                else:
                    success = False
            episode_buffer.append(transition(observation, activations, reward))
        config_buffer.append(configuration(episode, episode_buffer, success))
        episode_buffer = []
        observation = env.reset()
        done = False 
    print('Configuration Accuracy {}'.format(ep_solved/args.ep_per_config_test))
    return config_buffer, ep_solved

def test(args):
    model_info = namedtuple('model_info', ('configuration_file', 'configuration_buffer'))
    utils.set_seeds(args.seed)

    custom_objects_PPO = {
        "lr_schedule": lambda x: args.lr,
        "clip_range": lambda x: .2
    }
    model = PPO.load(args.model_out_dir + args.base_folder + args.base_file,\
        custom_objects = custom_objects_PPO, device = 'cpu')

    curr = args.curriculum
    print('Testing on {}'.format(curr['name']))
    model_buffer = []
    tot_ep_solved = 0
    for config in range(len(curr['test_files'])):
        print('Configuration {}'.format(config))
        config_file = curr['folder'] + curr['test_files'][config]
        if config == 0:
            env = AAI_utils.get_env(config_file, args, False)
        else:
            env._env.reset(arenas_configurations = ArenaConfig(config_file))
        config_buffer, ep_solved = test_config(env, model, args)
        tot_ep_solved += ep_solved
        model_buffer.append(model_info(config_file, config_buffer))
    env.close()

    print('Test Accuracy {}'.format(tot_ep_solved/(len(curr['test_files'])*args.ep_per_config_test)))

    df = pd.DataFrame.from_records(model_buffer)
    PATH = utils.check_path(args.activation_out_dir + args.base_folder + args.ckpt_folder)
    df.to_csv(PATH + args.base_file + '.csv')
    

        
            