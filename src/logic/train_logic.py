import torch 
from torch import nn
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

from src.logic import logic_utils
from src import utils

def train(model, args, train_data):
    # trains logic gate network
    writer = SummaryWriter(args.log_out_dir + args.base_folder + args.base_file) 
    dataloader = torch.utils.data.DataLoader(train_data, batch_size = args.batch_size, shuffle = True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    print('Training on {}'.format(args.dataset))
    
    for e in range(0, args.num_epochs):
        avg_accuracy, total, avg_loss = 0, 0, 0
        for _, data in enumerate(dataloader, 0):
            inputs, targets = data[0], data[1]
            optimizer.zero_grad()
            logits = model(inputs.float())
            avg_accuracy += logic_utils.accuracy(logits, targets)
            loss = criterion(logits, targets.long())
            avg_loss += loss.item()
            loss.backward()
            optimizer.step()
            total += 1 
        writer.add_scalar('Loss', (avg_loss/total), e)
        writer.add_scalar('Accuracy', (avg_accuracy/total), e)  
        if (e + 1) % 1000 == 0:
            print(f"Epoch {e + 1} Accuracy {avg_accuracy/total} Loss {avg_loss/total}")
    PATH = utils.check_path(args.model_out_dir + args.base_folder)
    torch.save(model, PATH + args.base_file) 
    writer.close()
    return model