from src.parsers import AAI_parser
from src.animalai import train_AAI, test_AAI
from src.PID import runner

if __name__ == "__main__":
    """
    ************************************************************
    Main function to run Animal AI experiments (for 1 seed). 
    ************************************************************
    """
    parser = AAI_parser.get_parser()  
    args = parser.parse_args()

    print("\nAnimal AI experiment: on {config}".\
                                format(config=args.curriculum['name']))
    
    # ---- Suggested naming ----
    # args.model_out_dir = 'trained_models/animalai/' 
    # args.activation_out_dir = 'activations/animalai/'
    # args.PID_out_dir = 'PID/animalai/'
    # args.log_out_dir = 'logs/animalai/'
    # args.base_folder = args.curriculum['name'] + '/'

    args.actor_layer_size = [args.actor_layer_size, args.actor_layer_size]
    args.critic_layer_size = [args.critic_layer_size, args.critic_layer_size]
    
    # Training
    
    train_AAI.train(args)

    # Testing -- folder/file naming depends on model checkpoint being tested
    # args.base_file = 'model_' + #seed + '_' + #num_steps
    # args.base_folder = args.base_folder + 'config_' + str(config) + '/'
    # args.ckpt_folder = '1T/' # or any ckpt name for num of steps
    test_AAI.test(args)

    # PID
    args.base_folder = args.base_folder + args.ckpt_folder
    runner.run_folder_models(args)


    # After running multiple seeds
    runner.run_batch(args)

