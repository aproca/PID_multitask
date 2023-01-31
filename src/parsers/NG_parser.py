import argparse
from src.configs import NG_curriculums

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
    for key, _ in NG_curriculums.__dict__.items():
        if str(key) == v:
            return NG_curriculums.__dict__[key]

def get_parser():
    parser = argparse.ArgumentParser(description='neurogym')

    # General
    parser.add_argument('--model_out_dir', type=str, default='trained_models/neurogym/' ,
                        help='Path to the output folder for saving the models.')
    parser.add_argument('--activation_out_dir', type=str, default='activations/neurogym/',
                        help='Path to the output folder for saving activations.')
    parser.add_argument('--PID_out_dir', type=str, default='PID/neurogym/',
                        help='Path to the output folder for PID measures.')
    parser.add_argument('--log_out_dir', type=str, default='logs/neurogym/',
                        help='Path to the output folder for logging.')
    parser.add_argument('--ckpt_folder', type=str, default='')
    parser.add_argument('--seed', type=int, default=0,
                        help='Model seed')

    # Experiment type
    parser.add_argument('--experiment', type=str, default='neurogym',
                        help='logic, animalai, neurogym')

    # Training hps
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--steps_per_config', type=int, default=40000,
                        help='Number of gradient steps to train on a single configuration')
    parser.add_argument('--lr', type=float, default=3e-3,
                        help='Learning rate')
    
    # Dataset
    parser.add_argument('--curriculum', type=str2config, default=NG_curriculums.cur_1,
                        help='Training/testing curriculum')
    parser.add_argument('--seq_len', type=int, default=100,
                        help='Length of input sequence for tasks')
    parser.add_argument('--interleaved', type=str2bool, default=False,
                        help='Whether to train with interleaving')

    # Main network config
    parser.add_argument('--hidden_size', type=int, default=10)

    # Testing
    parser.add_argument('--ep_per_config_test', type=int, default=50)

    # PID 
    parser.add_argument('--compute_PID', type=str2bool, default=True,
                        help='Whether to compute PID values (False to just get performance info)')
    parser.add_argument('--discrete_PID', type=str2bool, default=False,
                        help='Whether discrete or continuous PID measures computed')
    parser.add_argument('--network_type', type=str, default='recurrent',
                        help='feedforward, recurrent')
    parser.add_argument('--k_order', type=int, default=2,
                        help='K-order to compute PID')
    parser.add_argument('--full_order', type=str2bool, default=False,
                        help='Whether to compute full order measures')
    parser.add_argument('--compute_redundancy', type=str2bool, default=False,
                        help='Whether to compute redundancy measures')
    parser.add_argument('--efficient_korder', type=str2bool, default=True,
                        help='Samples 45 tuples if number of tuples exceeds 45')
    parser.add_argument('--save_korder_tuples', type=str2bool, default=False,
                        help='Whether to save individual korder tuple values')

    return parser