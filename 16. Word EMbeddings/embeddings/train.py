import os
import logging 

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from omegaconf import DictConfig, OmegaConf
import hydra

from model import W2VModel
from dataset import W2VCorpus, W2VDataset

logging.basicConfig()
logging.root.setLevel(logging.NOTSET)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def run(args):
    writer = SummaryWriter()
    filename = args.checkpoint_file
    file_path = args.file_path
    if os.path.isdir(file_path) is False:
        os.mkdir(file_path)

    corp = W2VCorpus(args.dataset_path)
    pairs = corp.make_positive_pairs()
    train_data = W2VDataset(pairs)
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size)
    loaders = {"train": train_dataloader}

    if torch.cuda.is_available and args.device_gpu:
        device = 'cuda'
    else:
        device = 'cpu'

    logger.info(f"Model will be trained on {device}")

    model = W2VModel(voc_size=len(corp.vocabulary), emb_size=300)
    model.to(device)
    optimzer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        running_loss = 0
        size = 0
        for batch_idx, data in enumerate(loaders['train']):
            input = data['word'].to(device)
            target = data['context'].to(device)
            
            optimzer.zero_grad()
            output = model.forward(input)
            loss = criterion(output, target)
            
            running_loss += loss.item()
            size += len(data)

            loss.backward()
            optimzer.step()
            
        avg_loss = running_loss / size
        
        writer.add_scalar("Loss/train", avg_loss, epoch)
        logger.debug(f"Loss: {avg_loss}")

    writer.flush()
    writer.close()
        
    torch.save(
        {
            "model": model.state_dict(),
            "optimzer": optimzer.state_dict(),
            "epoch": num_epoch,
            "loss": avg_loss
        }, 
        os.path.join(file_path, filename)
    )

@hydra.main(config_path="conf", config_name='config.yaml')
def main(args):
    run(args)

if __name__ == "__main__":
    main()