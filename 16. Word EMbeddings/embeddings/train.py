import numpy as np
from tqdm import tqdm
import logging 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import W2VCorpus, W2VDataset
from model import W2VModel
import os
logging.basicConfig()
logging.root.setLevel(logging.NOTSET)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

writer = SummaryWriter()

file_path = './output'
filename = "model_w2v.pt"
if os.path.isdir(file_path) is False:
    os.mkdir(file_path)

corp = W2VCorpus("text8")
pairs = corp.make_positive_pairs()
train_data = W2VDataset(pairs)
train_dataloader = DataLoader(train_data, batch_size=1024)
loaders = {"train": train_dataloader}

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
logger.info(f"Model will be trained on {device}")

model = W2VModel(voc_size=len(corp.vocabulary), emb_size=300)
model.to(device)
optimzer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
num_epoch = 1

for epoch in range(num_epoch):
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
    logger.debug(f"Loss: {avg_loss / (epoch + 1)}")

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
