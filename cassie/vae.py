import torch
from torch import nn, optim
from torch.nn import functional as F


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VAE(nn.Module):
    def __init__(self, hidden_size, latent_size, input_size, mj_state=False):
        super(VAE, self).__init__()

        if mj_state:
            self.input_size = input_size
        else:
            self.input_size = 40

        self.fc1 = nn.Linear(self.input_size, hidden_size)
        self.fc21 = nn.Linear(hidden_size, latent_size)
        self.fc22 = nn.Linear(hidden_size, latent_size)

        self.fc3 = nn.Linear(latent_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, self.input_size)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        # return torch.sigmoid(self.fc4(h3))
        return self.fc4(h3)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_size))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Should predict/reconstruct next step in sequence
# NOTE: Encoder and Decoder should be separate RNN's right? Meaning that when doing the full sequence
# of encoding then decoding hidden/cell states are not carried through encoding forward to decoding
# forward. So after encoding, when doing decoding foward hidden and cell states for decoder are reset
class RNN_VAE(nn.Module):
    def __init__(self, hidden_size, latent_size, device='cuda', mj_state=False):
        super(RNN_VAE, self).__init__()

        if mj_state:
            self.input_size = 33
        else:
            self.input_size = 40
        self.device = device

        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.encode_LSTM_hidden = torch.nn.LSTMCell(input_size=self.input_size, hidden_size=hidden_size)
        self.encode_LSTM_mu = torch.nn.LSTMCell(input_size=hidden_size, hidden_size=latent_size)
        self.encode_LSTM_logvar = torch.nn.LSTMCell(input_size=hidden_size, hidden_size=latent_size)

        self.decode_LSTM1 = torch.nn.LSTMCell(input_size=latent_size, hidden_size=hidden_size)
        self.decode_LSTM2 = torch.nn.LSTMCell(input_size=hidden_size, hidden_size=self.input_size)
        
        self.hx1 = None
        self.cx1 = None
        self.hx2 = None
        self.cx2 = None
        self.hx_hidden = None#torch.zeros(batch_size, self.hidden_size).to(self.device)
        self.cx_hidden = None#torch.zeros(batch_size, self.hidden_size).to(self.device)
        self.hx_mu = None#torch.zeros(batch_size, self.latent_size).to(self.device)
        self.cx_mu = None#torch.zeros(batch_size, self.latent_size).to(self.device)
        self.hx_logvar = None#torch.zeros(batch_size, self.latent_size).to(self.device)
        self.cx_logvar = None#torch.zeros(batch_size, self.latent_size).to(self.device)
        # self.mu_output = None#torch.zeros(batch_size, seq_len, self.latent_size).to(self.device)
        # self.logvar_output = None#torch.zeros(batch_size, seq_len, self.latent_size).to(self.device)
    # Assumes that "x" is a sequence of data of shape (batch_size, seq_len, feature_size)
    # Takes in a sequence of states, and will output encoded sequence
    def encode(self, x):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        
        # # Initialize hidden and cell state to be zero (NOTE: could try init as random as well)
        # hx_hidden = torch.zeros(batch_size, self.hidden_size).to(self.device)
        # cx_hidden = torch.zeros(batch_size, self.hidden_size).to(self.device)
        # hx_mu = torch.zeros(batch_size, self.latent_size).to(self.device)
        # cx_mu = torch.zeros(batch_size, self.latent_size).to(self.device)
        # hx_logvar = torch.zeros(batch_size, self.latent_size).to(self.device)
        # cx_logvar = torch.zeros(batch_size, self.latent_size).to(self.device)
        mu_output = torch.zeros(batch_size, seq_len, self.latent_size).to(self.device)
        logvar_output = torch.zeros(batch_size, seq_len, self.latent_size).to(self.device)

        # Loop through sequence
        for i in range(seq_len):
            # print(i)
            self.hx_hidden, self.cx_hidden = self.encode_LSTM_hidden(x[:, i, :], (self.hx_hidden, self.cx_hidden))
            self.hx_mu, self.cx_mu = self.encode_LSTM_mu(self.hx_hidden, (self.hx_mu, self.cx_mu))
            self.hx_logvar, self.cx_logvar = self.encode_LSTM_logvar(self.hx_hidden, (self.hx_logvar, self.cx_logvar))
            mu_output[:, i, :] = self.hx_mu
            logvar_output[:, i, :] = self.hx_logvar
       
        return mu_output, logvar_output

    # Given the mean and log variance for a sequence of states, returns a sample from
    # the distribution at each timestep
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar) # 0.5 because stddev is square root of var 
        eps = torch.randn_like(std)
        # NOTE: I think does does mu + std scaled normal tensor because this is faster than making
        # a torch dist for every single element, and should do the same thing
        return mu + eps*std

    # Decodes a sequence of latent states back into a sequence of reconstructed states
    # Similar to encode function, assumes z is of shape (batch_size, seq_len, latent_size)
    def decode(self, z):
        batch_size = z.shape[0]
        seq_len = z.shape[1]
        
        # # Initialize hidden and cell state to be zero (NOTE: could try init as random as well)
        # hx1 = torch.zeros(batch_size, self.hidden_size).to(device)
        # cx1 = torch.zeros(batch_size, self.hidden_size).to(device)
        # hx2 = torch.zeros(batch_size, self.input_size).to(device)
        # cx2 = torch.zeros(batch_size, self.input_size).to(device)

        decode_output = torch.zeros(batch_size, seq_len, self.input_size).to(self.device)
        # Loop through sequence
        for i in range(seq_len):
            self.hx1, self.cx1 = self.decode_LSTM1(z[:, i, :], (self.hx1, self.cx1))
            self.hx2, self.cx2 = self.decode_LSTM2(self.hx1, (self.hx2, self.cx2))
            decode_output[:, i, :] = self.hx2
       
        return decode_output

    def reset_hidden(self, batch_size):
        # Initialize hidden and cell state to be zero (NOTE: could try init as random as well)
        self.hx1 = torch.zeros(batch_size, self.hidden_size).to(self.device)
        self.cx1 = torch.zeros(batch_size, self.hidden_size).to(self.device)
        self.hx2 = torch.zeros(batch_size, self.input_size).to(self.device)
        self.cx2 = torch.zeros(batch_size, self.input_size).to(self.device)
        # Initialize hidden and cell state to be zero (NOTE: could try init as random as well)
        self.hx_hidden = torch.zeros(batch_size, self.hidden_size).to(self.device)
        self.cx_hidden = torch.zeros(batch_size, self.hidden_size).to(self.device)
        self.hx_mu = torch.zeros(batch_size, self.latent_size).to(self.device)
        self.cx_mu = torch.zeros(batch_size, self.latent_size).to(self.device)
        self.hx_logvar = torch.zeros(batch_size, self.latent_size).to(self.device)
        self.cx_logvar = torch.zeros(batch_size, self.latent_size).to(self.device)

    def forward(self, x):
        # print(x.size())
        mu_seq, logvar_seq = self.encode(x)
        z = self.reparameterize(mu_seq, logvar_seq)
        return self.decode(z), mu_seq, logvar_seq


class RNN_VAE_FULL(nn.Module):
    def __init__(self, hidden_size, latent_size, num_layers, input_size, device="cuda", mj_state=False):
        super(RNN_VAE_FULL, self).__init__()

        if mj_state:
            self.input_size = input_size
        else:
            self.input_size = 40

        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.device= device
        self.encode_LSTM = torch.nn.LSTM(input_size=self.input_size, hidden_size=hidden_size, batch_first=True, num_layers=num_layers)
        self.encode_mu = nn.Linear(hidden_size, latent_size)
        self.encode_logvar = nn.Linear(hidden_size, latent_size)

        self.decode_LSTM = torch.nn.LSTM(input_size=latent_size, hidden_size=self.input_size, batch_first=True, num_layers=num_layers)

        self.he0 = None
        self.ce0 = None
        self.hd0 = None
        self.cd0 = None

    # Assumes that "x" is a sequence of data of shape (batch_size, seq_len, feature_size)
    # Takes in a sequence of states, and will output encoded sequence
    def encode(self, x):

        [batch_size, seq_len, _] = x.shape

        lstm_output, (_, _) = self.encode_LSTM(x, (self.he0, self.ce0))

        return self.encode_mu(lstm_output), self.encode_logvar(lstm_output)
       

    # Given the mean and log variance for a sequence of states, returns a sample from
    # the distribution at each timestep
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar) # 0.5 because stddev is square root of var 
        eps = torch.randn_like(std)
        # NOTE: I think does does mu + std scaled normal tensor because this is faster than making
        # a torch dist for every single element, and should do the same thing
        return mu + eps*std

    # Decodes a sequence of latent states back into a sequence of reconstructed states
    # Similar to encode function, assumes z is of shape (batch_size, seq_len, latent_size)
    def decode(self, z):
        
        [batch_size, seq_len, _] = z.shape

        decode_output, (_, _) = self.decode_LSTM(z, (self.hd0, self.cd0))

        return decode_output

    def reset_hidden(self, batch_size):
        self.he0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        self.ce0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)

        self.hd0 = torch.zeros(self.num_layers, batch_size, self.input_size).to(self.device)
        self.cd0 = torch.zeros(self.num_layers, batch_size, self.input_size).to(self.device)

    def forward(self, x):
        # print(x.size())
        mu_seq, logvar_seq = self.encode(x)
        z = self.reparameterize(mu_seq, logvar_seq)
        return self.decode(z), mu_seq, logvar_seq



class VAE_output_dist(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(VAE_output_dist, self).__init__()

        self.input_size = input_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc21 = nn.Linear(hidden_size, latent_size)
        self.fc22 = nn.Linear(hidden_size, latent_size)

        self.fc3 = nn.Linear(latent_size, hidden_size)
        self.fc41 = nn.Linear(hidden_size, input_size)
        self.fc42 = nn.Linear(hidden_size, input_size)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        # return torch.sigmoid(self.fc4(h3))
        return self.fc41(h3), self.fc42(h3)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_size))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class VAE_LSTM_FF(nn.Module):
    def __init__(self, hidden_size, latent_size, num_layers, input_size, device="cuda", mj_state=False):
        super(VAE_LSTM_FF, self).__init__()

        if mj_state:
            self.input_size = input_size
        else:
            self.input_size = 40

        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.device= device

        # LSTM
        self.encode_LSTM = torch.nn.LSTM(input_size=self.input_size, hidden_size=self.input_size, batch_first=True, num_layers=num_layers)
        self.encode_mu = nn.Sequential(
                        nn.Linear(self.input_size, self.hidden_size),
                        nn.Linear(self.hidden_size, 32),
                        nn.Linear(32, self.latent_size))
        self.encode_logvar = nn.Sequential(
                        nn.Linear(self.input_size, self.hidden_size),
                        nn.Linear(self.hidden_size, 32),
                        nn.Linear(32, self.latent_size))

        self.decode_LSTM = torch.nn.LSTM(input_size=latent_size, hidden_size=self.input_size, batch_first=True, num_layers=num_layers)

        self.he0 = None#torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        self.ce0 = None#torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

        self.hd0 = None#torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        self.cd0 = None#torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

    # Assumes that "x" is a sequence of data of shape (batch_size, seq_len, feature_size)
    # Takes in a sequence of states, and will output encoded sequence
    def encode(self, x):

        [batch_size, seq_len, _] = x.shape

        lstm_output, (_, _) = self.encode_LSTM(x, (self.he0, self.ce0))

        return self.encode_mu(lstm_output), self.encode_logvar(lstm_output)
       
    # Given the mean and log variance for a sequence of states, returns a sample from
    # the distribution at each timestep
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar) # 0.5 because stddev is square root of var 
        eps = torch.randn_like(std)
        # NOTE: I think does does mu + std scaled normal tensor because this is faster than making
        # a torch dist for every single element, and should do the same thing
        return mu + eps*std

    # Decodes a sequence of latent states back into a sequence of reconstructed states
    # Similar to encode function, assumes z is of shape (batch_size, seq_len, latent_size)
    def decode(self, z):
        
        [batch_size, seq_len, _] = z.shape

        decode_output, (_, _) = self.decode_LSTM(z, (self.hd0, self.cd0))

        return decode_output

    def reset_hidden(self, batch_size):
        self.he0 = torch.zeros(self.num_layers, batch_size, self.input_size).to(self.device)
        self.ce0 = torch.zeros(self.num_layers, batch_size, self.input_size).to(self.device)

        self.hd0 = torch.zeros(self.num_layers, batch_size, self.input_size).to(self.device)
        self.cd0 = torch.zeros(self.num_layers, batch_size, self.input_size).to(self.device)

    def forward(self, x):
        # print(x.size())
        mu_seq, logvar_seq = self.encode(x)
        z = self.reparameterize(mu_seq, logvar_seq)
        return self.decode(z), mu_seq, logvar_seq
