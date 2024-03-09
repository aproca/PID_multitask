import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_parser():
    parser = argparse.ArgumentParser(description='logicgate')

    # General
    parser.add_argument('--model_out_dir', type=str, default='trained_models/logic/COPY/',
                        help='Path to the output folder for saving the models.')
    parser.add_argument('--activation_out_dir', type=str, default='activations/logic/COPY/',
                        help='Path to the output folder for saving activations.')
    parser.add_argument('--PID_out_dir', type=str, default='PID/logic/COPY/',
                        help='Path to the output folder for PID measures.')
    parser.add_argument('--log_out_dir', type=str, default='logs/logic/COPY/',
                        help='Path to the output folder for tensorboarc logging.')
    parser.add_argument('--ckpt_folder', type=str, default='')
    parser.add_argument('--num_seeds', type=int, default=10,
                        help='Num model seeds to perform experiment on')
    parser.add_argument('--seed', type=int, default=0,
                        help='Model seed')

    # Experiment type
    parser.add_argument('--experiment', type=str, default='logic',
                        help='logic, animalai, neurogym')

    # Training hps
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Train batch size')
    parser.add_argument('--num_epochs', type=int, default=15000,
                        help='Num training epochs')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Training learning rate')

    # Main network config
    parser.add_argument('--hidden_size', type=int, default=10,
                        help='Hidden layer size')

    # Dataset
    parser.add_argument('--dataset', type=str, default='COPY',
                    help='COPY, XOR')
    parser.add_argument('--input_size', type=int, default=2,
                        help='Size of observation input')
    parser.add_argument('--output_size', type=int, default=2,
                        help='Size of action space')

    # PID 
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
    parser.add_argument('--IQR_bins', type=str2bool, default=False,
                        help='Whether to bin based on IQR')
    parser.add_argument('--k_order', type=int, default=2,
                        help='K-order to compute PID')
    parser.add_argument('--full_order', type=str2bool, default=True,
                        help='Whether to compute full order measures')
    parser.add_argument('--compute_redundancy', type=str2bool, default=True,
                        help='Whether to compute redundancy measures')
    parser.add_argument('--efficient_korder', type=str2bool, default=False,
                        help='Samples 45 tuples if number of tuples exceeds 45')
    parser.add_argument('--save_korder_tuples', type=str2bool, default=True,
                        help='Whether to save individual korder tuple values')

    return parser