import torch
from src.logic import logic_utils, train_logic, test_logic, lesion
from src.models.feedforward import FFNetwork
from src.PID import runner
from src import utils
from src.parsers import logic_parser

if __name__ == "__main__":
    """
    *****************************************************************
    Main function to run logic gate experiments (for multiple seeds).
    *****************************************************************
    """
    parser = logic_parser.get_parser()  
    args = parser.parse_args()

    print("\nLogic gate experiment: on {dataset}".\
                                format(dataset=args.dataset))
    
    # ---- Suggested naming example ----
    # args.model_out_dir = 'trained_models/logic/COPY/hidden_10/' 
    # args.activation_out_dir = 'activations/logic/COPY/hidden_10/'
    # args.PID_out_dir = 'PID/logic/COPY/hidden_10/'
    # args.log_out_dir = 'logs/logic/COPY/hidden_10/'

    args.layer_sizes = [args.hidden_size, args.hidden_size]
    seeds = list(range(0, args.num_seeds))
    dropout_p = [0.0, 0.1, 0.3, 0.5]
    data = logic_utils.get_data(args)

    for p in dropout_p:
        args.base_folder = str(p) + '/'
        for s in seeds:
            utils.set_seeds(s)
            args.base_file = str(s)
            model = FFNetwork(args, p, lesion = False)
            train_logic.train(model, args, data)
            model = torch.load(args.model_out_dir + args.base_folder + args.base_file)
            test_logic.test(model, args, data)

    for p in dropout_p: 
        args.base_folder = str(p) + '/'
        runner.run_folder_models(args)
        runner.run_batch(args)

    args.compute_PID = False
    lesion.lesion_experiment(args, dropout_p, seeds, data)