from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from cassie.vae import VAE

import numpy as np
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='VAE Cassie')
parser.add_argument('--batch_size', type=int, default=2048, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--latent_size', type=int, default=20, help='size of latent space')
parser.add_argument('--hidden_size', type=int, default=40, help='size of hidden space')
parser.add_argument('--test_model', type=str, default=None, help='path to model to load')
parser.add_argument('--run_name', type=str, default=None, help='name of model to save and associated log data')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")
now = datetime.now()

if args.test_model is None:
    log_path = "./logs/"+args.run_name
    logger = SummaryWriter(log_path, flush_secs=0.1) # flush_secs=0.1 actually slows down quite a bit, even on parallelized set ups

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

dataset_np = np.load("./5b75b3-seed0_full_mjdata/total_data.npy")
X_train, X_test = train_test_split(dataset_np, test_size=0.05, random_state=42, shuffle=True)

# remove clock and speed commands, only dynamics
X_train = X_train[:, 0:-3]
X_test = X_test[:, 0:-3]

input_dim = 35 + 32 - 3

data_min = np.min(X_train, axis=0)
data_max = np.max(X_train-data_min, axis=0)
norm_data = np.divide((X_train-data_min), data_max)
norm_test_data = np.divide((X_test-data_min), data_max)

norm_data = torch.Tensor(norm_data)
norm_test_data = torch.Tensor(norm_test_data)
# norm_data = torch.Tensor(X_train)
# norm_test_data = torch.Tensor(X_test)
# print()

np.savez("./data_norm_params.npz", data_max=data_max, data_min=data_min)
# exit()

data_max = torch.Tensor(data_max).to(device)
data_min = torch.Tensor(data_min).to(device)

model = VAE(input_dim, args.hidden_size, args.latent_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
recon_loss_cri = nn.MSELoss()

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    MSE = input_dim * recon_loss_cri(recon_x, x.view(-1, input_dim))

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE + KLD

def train(epoch):
    model.train()
    train_loss = 0
    train_len = len(norm_data)

    sampler = BatchSampler(SubsetRandomSampler(range(train_len)), args.batch_size, drop_last=True)

    for batch_idx, indices in enumerate(sampler):
        data = norm_data[indices]
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), train_len,
                100. * batch_idx / args.batch_size,
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / train_len))
    logger.add_scalar("Train/Loss", train_loss / train_len, epoch)
    for name, param in model.named_parameters():
        logger.add_histogram("Model Params/"+name, param.data, epoch)

def test(epoch):
    model.eval()
    test_loss = 0

    with torch.no_grad():
        test_len = len(norm_test_data)

        sampler = BatchSampler(SubsetRandomSampler(range(test_len)), args.batch_size, drop_last=True)

        for batch_idx, indices in enumerate(sampler):
            data = norm_data[indices]
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            # Un-normalize data
            orig_batch = recon_batch*data_max + data_min
            orig_data = data*data_max + data_min
            percent_error = torch.div((orig_data-orig_batch), (orig_data+1e-6))
            percent_error = torch.mean(percent_error, axis=0)
            if batch_idx == 0:
                # print(orig_batch[0,:])
                # print(orig_data[0,:])
                print("percent error: ", percent_error*100)
            
    test_loss /= test_len
    print('====> Test set loss: {:.4f}'.format(test_loss))
    logger.add_scalar("Test/Loss", test_loss, epoch)

# if __name__ == "__main__":

if args.test_model is None:
    # train and save new model
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)    
    PATH = "./vae_model/"+args.run_name+".pt"
    torch.save(model.state_dict(), PATH)

# Test trained model (or loaded model)
print("Testing model.")
if args.test_model is not None:
    PATH = args.test_model
model.cpu()
model.load_state_dict(torch.load(PATH))
model.eval()
print(model)
print()
recon_x, mu, logvar = model(norm_data[0,:])
print("reconstructed data:")
print(recon_x)
print()
print("normalized test data:")
print(norm_data[0,:])
