from src.parsers import NG_parser
from src.neurogym import train_NG, test_NG 
from src.PID import runner

if __name__ == "__main__":
    """
    ************************************************************
    Main function to run Neurogym experiments (for 1 seed). 
    ************************************************************
    """
    parser = NG_parser.get_parser() 
    args = parser.parse_args()

    print("\nNeurogym experiment: on {config}".\
                                format(config=args.curriculum['name']))

    # ---- Suggested naming ----
    # args.model_out_dir = 'trained_models/neurogym/' 
    # args.activation_out_dir = 'activations/neurogym/'
    # args.PID_out_dir = 'PID/neurogym/'
    # args.log_out_dir = 'logs/neurogym/'
    
    args.base_folder = args.curriculum['name'] + '/'
    if args.interleaved:
        args.base_folder = args.base_folder[:-1] + 'I/'
    
    # Training
    
    train_NG.train(args)

    # Testing -- folder/file naming depends on model checkpoint being tested
    # args.base_folder = args.base_folder + 'config_' + str(config) + '/'
    # args.base_file = 'model_' + #seed + '_' + #num_steps
    # args.ckpt_folder = '1T/' # or any ckpt name for num of steps
    test_NG.test(args)

    # PID
    args.base_folder = args.base_folder + args.ckpt_folder
    runner.run_folder_models(args)
    
    # After running multiple seeds
    runner.run_batch(args)
