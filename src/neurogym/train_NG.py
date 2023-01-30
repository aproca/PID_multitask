import numpy as np
import torch
from torch import nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from src.models.recurrent import RecurrentNetwork
from src.neurogym import NG_utils
from src import utils

def initialize_model(args, dataset):
    model = RecurrentNetwork(dataset.env.observation_space.shape[0] + 1, dataset.env.action_space.n, args)
    PATH = utils.check_path(args.model_out_dir + args.base_folder + 'config_0/')
    torch.save(model, '{}model_{}_0'.format(PATH, args.seed))
    weights = [20.]*(dataset.env.action_space.n - 1)
    weights.insert(0, 1.)
    weights = torch.tensor(weights)
    criterion = nn.CrossEntropyLoss(weights)
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    return model, criterion, optimizer

def run_epoch(model, args, dataset, curr, config, optimizer, criterion):
    metrics = dict()
    optimizer.zero_grad()
    inputs, labels = NG_utils.add_task_indicator(dataset, curr['indicators'][config])
    predictions = model(inputs)
    decision_timesteps = np.where(inputs[:,:,0]==0)
    format_labels = torch.from_numpy((labels).flatten()).type(torch.long)
    loss = criterion(predictions.view(-1, dataset.env.action_space.n), format_labels)
    actions = np.argmax(predictions.detach().clone().numpy(), axis = -1)
    metrics['decision_error'] = len(np.where(labels[decision_timesteps] != actions[decision_timesteps])[0])
    metrics['decision_total'] = len(decision_timesteps[0])
    metrics['total_correct'] = (actions == labels).sum().item()
    metrics['total_actions'] = args.seq_len*args.batch_size 
    metrics['avg_loss'] = loss.item()
    loss.backward()
    optimizer.step()
    return metrics

def run_sequential(args, curr, model, datasets, optimizer, criterion):
    # trains configurations sequentially
    for config in range(len(curr['config_files'])):
        print('Configuration {}'.format(config))
        writer = SummaryWriter('{}{}config_{}/'.format(args.log_out_dir, args.base_folder, config))
        metrics = dict()
        for epoch in range(0, args.steps_per_config):
            epoch_metrics = run_epoch(model, args, datasets[config], curr, config, optimizer, criterion)           

            if len(metrics) > 0 and len(epoch_metrics) > 0:
                for key, value in epoch_metrics.items():
                    metrics[key] += value
            else:
                for key, value in epoch_metrics.items():
                    metrics[key] = value

            if (epoch + 1) % 200 == 0:
                print('Loss ', metrics['avg_loss']/200)
                print('Accuracy ', metrics['total_correct']/metrics['total_actions'])
                print('Decision Accuracy ', (metrics['decision_total'] - \
                    metrics['decision_error'])/metrics['decision_total'])
                writer.add_scalar('Loss', (metrics['avg_loss']/200), epoch)
                writer.add_scalar('Accuracy', (metrics['total_correct']/metrics['total_actions']), epoch)
                writer.add_scalar('Decision Accuracy', (metrics['decision_total'] - \
                    metrics['decision_error'])/metrics['decision_total'], epoch)
                metrics = dict()

        PATH = utils.check_path('{}{}config_{}/'.format(args.model_out_dir, args.base_folder, config))
        torch.save(model, '{}model_{}_{}'.format(PATH, args.seed, args.steps_per_config))
        writer.close()

def run_interleaved(args, curr, model, datasets, optimizer, criterion):
    # trains by interleaving configurations
    writer = SummaryWriter('{}{}config_0/'.format(args.log_out_dir, args.base_folder))
    metrics = dict()
    count = 0
    print('Interleaved')
    for epoch in range(0, args.steps_per_config): 
        for config in range(0, len(curr['config_files'])):
            epoch_metrics = run_epoch(model, args, datasets[config], curr, config, optimizer, criterion)

            if len(metrics) > 0 and len(epoch_metrics) > 0 and count == len(curr['config_files']):
                for key, value in epoch_metrics.items():
                    metrics[str(config) + key] += value
            else:
                for key, value in epoch_metrics.items():
                    metrics[str(config) + key] = value
                count = config

            if (epoch + 1) % 100 == 0:
                print('Task {} Loss {}'.format(config, (metrics['{}avg_loss'.format(config)]/100)))
                print('Task {} Accuracy {}'.format(config, (metrics['{}total_correct'.format(config)]/metrics['{}total_actions'.format(config)])))
                print('Task {} Decision Accuracy {}'.format(config, (metrics['{}decision_total'.format(config)] - \
                    metrics['{}decision_error'.format(config)])/metrics['{}decision_total'.format(config)]))
                writer.add_scalar('Task {} Loss {}'.format(config, (metrics['{}avg_loss'.format(config)]/100), len(curr['config_files'])*epoch))
                writer.add_scalar('Task {} Accuracy {}'.format(config, (metrics['{}total_correct'.format(config)]/metrics['{}total_actions'.format(config)]),\
                    len(curr['config_files'])*epoch))
                writer.add_scalar('Task {} Decision Accuracy {}'.format(config, (metrics['{}decision_total'.format(config)] - \
                    metrics['{}decision_error'.format(config)])/metrics['{}decision_total'.format(config)], len(curr['config_files'])*epoch))

        if (epoch + 1) % 100 == 0:
            all_metrics = dict()
            for key, value in metrics.items():
                if key[0] == '0':
                    all_metrics[key[1:]] = value
                    for config in range(1, len(curr['config_files'])):
                        all_metrics[key[1:]] += metrics[str(config) + key[1:]]

            print('Loss', (all_metrics['avg_loss']/100))
            print('Accuracy', (all_metrics['total_correct']/all_metrics['total_actions']))
            print('Decision Accuracy', (all_metrics['decision_total'] - \
                all_metrics['decision_error'])/all_metrics['decision_total'])
            writer.add_scalar('Loss', (all_metrics['avg_loss']/100), len(curr['config_files'])*epoch)
            writer.add_scalar('Accuracy', (all_metrics['total_correct']/all_metrics['total_actions']),\
                len(curr['config_files'])*epoch)
            writer.add_scalar('Decision Accuracy', (all_metrics['decision_total'] - \
                all_metrics['decision_error'])/all_metrics['decision_total'], len(curr['config_files'])*epoch)
            metrics = dict()
        
        if (epoch + 1) % int(args.steps_per_config/len(curr['config_files'])) == 0:
            PATH = utils.check_path('{}{}config_0/'.format(args.model_out_dir, args.base_folder))
            torch.save(model, '{}model_{}_{}'.format(PATH, args.seed, int((epoch + 1) * len(curr['config_files']))))

    PATH = utils.check_path('{}{}config_0/'.format(args.model_out_dir, args.base_folder))
    torch.save(model, '{}model_{}_'.format(PATH, args.seed, int(args.steps_per_config*len(curr['config_files']))))
    writer.close()

def train(args):
    utils.set_seeds(args.seed)
    curr = args.curriculum
    datasets = dict()
    for config in range(len(curr['config_files'])):
        datasets[config] = NG_utils.load_dataset(curr['config_files'][config], args)

    model, criterion, optimizer = initialize_model(args, datasets[0])
    print('Training on {}'.format(curr['name']))

    if args.interleaved:
        run_interleaved(args, curr, model, datasets, optimizer, criterion)
    else:
        run_sequential(args, curr, model, datasets, optimizer, criterion)



        


             

        
            
            

    