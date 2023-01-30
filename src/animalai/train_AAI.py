from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from animalai.envs.arena_config import ArenaConfig

from src import utils
from src.animalai import AAI_utils
from src.models.actor_critic import NetActorCriticPolicy

def train(args):
    utils.set_seeds(args.seed)
    curr = args.curriculum
    print('Training on {}'.format(curr['name']))
    num_epochs = int(args.steps_per_config/args.save_freq)

    for config in range(len(curr['train_files'])):
        config_file = curr['folder'] + curr['train_files'][config]
        print('Configuration {}'.format(config))
        reset_timesteps = True
        if config == 0:
            env = AAI_utils.get_env(config_file, args, True)
            eval_env = AAI_utils.get_env(config_file, args, True)
            eval_callback = EvalCallback(eval_env, best_model_save_path =\
                args.model_out_dir + args.base_folder + 'config_{}/'.format(config), eval_freq = args.save_freq,
                n_eval_episodes = 48, deterministic = True, render = False)
            model = PPO(NetActorCriticPolicy, env, verbose = args.verbose,\
                device = 'cpu', policy_kwargs = dict(net_arch=dict(pi=args.actor_layer_size,\
                vf=args.critic_layer_size)), seed = args.seed, learning_rate = args.lr,
                tensorboard_log = args.log_out_dir + args.base_folder + 'config_{}/'.format(config))
            model.save(args.model_out_dir + args.base_folder + 'config_{}/model_{}_0'.format(config, args.seed))
        else:
            env._env.reset(arenas_configurations = ArenaConfig(config_file))
            model.set_env(env)

        for epoch in range(num_epochs):
            model = model.learn(args.save_freq, reset_num_timesteps = reset_timesteps,\
                callback = eval_callback, n_eval_episodes = 48)
            model.save(args.model_out_dir + args.base_folder + 'config_{}/model_{}_{}'.format(config,\
                args.seed, (epoch+1)*args.save_freq))
            reset_timesteps = False
            if curr['thresholds'][config] < 100:
                mean_reward, _ = evaluate_policy(model, env)
                if mean_reward >= curr['thresholds'][config]:
                    break
    
    env.close()
    eval_env.close()


            
