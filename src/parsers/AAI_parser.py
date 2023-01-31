import argparse
from src.configs import AAI_curriculums

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2config(v):
    for k, v in AAI_curriculums.__dict__.items():
        if str(k) == v:
            return AAI_curriculums.__dict__[k]

def get_parser():
    parser = argparse.ArgumentParser(description='animalai')

    # General
    parser.add_argument('--model_out_dir', type=str, default='trained_models/animalai/' ,
                        help='Path to the output folder for saving the models.')
    parser.add_argument('--activation_out_dir', type=str, default='activations/animalai/',
                        help='Path to the output folder for saving activations.')
    parser.add_argument('--PID_out_dir', type=str, default='PID/animalai/',
                        help='Path to the output folder for PID measures.')
    parser.add_argument('--log_out_dir', type=str, default='logs/animalai/',
                        help='Path to the output folder for logging.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Model seed')

    # Dataset
    parser.add_argument('--curriculum', type=str2config, default=AAI_curriculums.cur_1,
                        help='Training/testing curriculum')
    
    # Experiment type
    parser.add_argument('--experiment', type=str, default='animalai',
                        help='logic, animalai, neurogym')

    # Main network config
    parser.add_argument('--hidden_size', type=int, default=10,
                        help='Hidden layer size')
    parser.add_argument('--actor_layer_size', type=int, default=10,
                        help='Hidden layer sizes for actor network')
    parser.add_argument('--critic_layer_size', type=int, default=10,
                        help='Hidden layer sizes for critic network')

    # AAI
    parser.add_argument('--frame_skips', type=int, default=1,
                        help='Whether to use frame skips when training')
    parser.add_argument('--save_freq', type=int, default=10000,
                        help='Save frequency of model during training')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Train logging output')

    # Training
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--steps_per_config', type=int, default=1000000,
                        help='Number of steps to train for a single configuration')

    # Testing
    parser.add_argument('--ep_per_config_test', type=int, default=1,
                        help='Number of episodes to test on for a configuration.')

    # PID params
    parser.add_argument('--compute_PID', type=str2bool, default=True,
                        help='Whether to compute PID values (False to just get performance info)')
    parser.add_argument('--discrete_PID', type=str2bool, default=True,
                        help='Whether discrete or continuous PID measures computed')
    parser.add_argument('--network_type', type=str, default='feedforward',
                        help='feedforward, recurrent')
    parser.add_argument('--layer_max_bin', type=int, default=5,
                        help='Maximum bin size for layerwise discretization')
    parser.add_argument('--layer_num_bins', type=int, default=3,
                        help='Maximum bin size for layerwise discretization')
    parser.add_argument('--k_order', type=int, default=2,
                        help='K-order to compute PID')
    parser.add_argument('--compute_redundancy', type=str2bool, default=False,
                        help='Whether to compute redundancy measures')
    parser.add_argument('--efficient_korder', type=str2bool, default=True,
                        help='Samples 45 tuples if number of tuples exceeds 45')
    parser.add_argument('--save_korder_tuples', type=str2bool, default=False,
                        help='Whether to save individual korder tuple values')
    parser.add_argument('--full_order', type=str2bool, default=False,
                        help='Whether to compute full order measures')


    # AAI raycast observation discretization
    parser.add_argument('--ray_max_bin', type=int, default=1,
                        help='Maximum bin size for AAI raycast observation')
    parser.add_argument('--ray_num_bins', type=int, default=3,
                        help='Number of bins for AAI raycast observation')
    parser.add_argument('--pos_max_bin', type=int, default=40,
                        help='Maximum bin size for AAI raycast observation')
    parser.add_argument('--pos_num_bins', type=int, default=5,
                        help='Number of bins for AAI raycast observation')

    return parser