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

import numpy as np
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='VAE Cassie')
parser.add_argument('--batch_size', type=int, default=2048, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")
now = datetime.now()
log_path = "./logs/"+now.strftime("%Y%m%d-%H%M%S")
logger = SummaryWriter(log_path, flush_secs=0.1) # flush_secs=0.1 actually slows down quite a bit, even on parallelized set ups

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

dataset_np = np.load("./5b75b3-seed0_gen_data/total_data.npy")
# dataset_np = torch.Tensor(dataset_np)
X_train, X_test = train_test_split(dataset_np, test_size=0.05, random_state=42, shuffle=True)

data_min = np.min(X_train, axis=0)
data_max = np.max(X_train-data_min, axis=0)
norm_data = np.divide((X_train-data_min), data_max)
norm_test_data = np.divide((X_test-data_min), data_max)

norm_data = torch.Tensor(norm_data)
norm_test_data = torch.Tensor(norm_test_data)

print("norm data: ", norm_data.shape)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(43, 100)
        self.fc21 = nn.Linear(100, 100)
        self.fc22 = nn.Linear(100, 100)

        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, 43)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 43))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 43), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

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
            # if i == 0:
            #     n = min(data.size(0), 8)
            #     comparison = torch.cat([data[:n],
            #                           recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
            #     save_image(comparison.cpu(),
            #              './results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= test_len
    print('====> Test set loss: {:.4f}'.format(test_loss))
    logger.add_scalar("Test/Loss", test_loss, epoch)
 
if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        # with torch.no_grad():
        #     sample = torch.randn(64, 20).to(device)
        #     sample = model.decode(sample).cpu()
        #     save_image(sample.view(64, 1, 28, 28),
        #                './results/sample_' + str(epoch) + '.png')