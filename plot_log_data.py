import numpy as np
import matplotlib.pyplot as plt
import math

def plot_FF_data():
    # Load data
    latent_sizes = [2, 5, 10, 15, 20, 25]

    test_loss = []
    KL_loss = []

    for i in range(len(latent_sizes)):
        test = np.genfromtxt("./plot_data/FF/run-mj_state_SSE_KL_NoXY_500epoch_latent_{}_hidden_40-tag-Test_Loss.csv".format(latent_sizes[i]), 
                delimiter=",", skip_header=1, usecols=(2))
        KL = np.genfromtxt("./plot_data/FF/run-mj_state_SSE_KL_NoXY_500epoch_latent_{}_hidden_40-tag-Train_Loss_KL.csv".format(latent_sizes[i]), 
                delimiter=",", skip_header=1, usecols=(2))
        test_loss.append(test)
        KL_loss.append(KL)

    fig, ax = plt.subplots(figsize=(7, 5))
    for i in range(len(latent_sizes)):
        ax.plot(test_loss[i], label="Latent Size {}".format(latent_sizes[i]))
    ax.set_title("Testing Loss Comparison for Different Latent Sizes")
    ax.set_ylabel("Test Loss")
    ax.set_xlabel("Epoch")
    ax.set_yscale('log')
    ax.legend()
    plt.savefig("./FF_SSE_KL_NoXY_test_loss.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    for i in range(len(latent_sizes)):
        ax.plot(KL_loss[i], label="Latent Size {}".format(latent_sizes[i]))
    ax.set_title("KL Divergence Loss Comparison for Different Latent Sizes")
    ax.set_ylabel("KL Div Loss")
    ax.set_xlabel("Epoch")
    ax.legend()
    # plt.show()
    plt.savefig("./FF_SSE_KL_NoXY_KL_loss.png")

def plot_LSTM_layer_sweep():
    layers = [1, 3, 5]
    latent_size = [2, 15, 25]
    latent2_data = []
    latent15_data = []
    latent25_data = []
    for i in range(len(layers)):
        data = np.genfromtxt("./plot_data/LSTM_layer_sweep/run-mj_state_lstm_SSE_NoKL_NoXY_latent_2_layers_{}_hidden_40-tag-Test_Loss.csv".format(
                    layers[i]), delimiter=",", skip_header=1, usecols=(2))
        latent2_data.append(data)
    for i in range(len(layers)):
        data = np.genfromtxt("./plot_data/LSTM_layer_sweep/run-mj_state_lstm_SSE_NoKL_NoXY_latent_15_layers_{}_hidden_40-tag-Test_Loss.csv".format(
                    layers[i]), delimiter=",", skip_header=1, usecols=(2))
        latent15_data.append(data)
    for i in range(len(layers)):
        data = np.genfromtxt("./plot_data/LSTM_layer_sweep/run-mj_state_lstm_SSE_NoKL_NoXY_latent_25_layers_{}_hidden_40-tag-Test_Loss.csv".format(
                    layers[i]), delimiter=",", skip_header=1, usecols=(2))
        latent25_data.append(data)

    fig, ax = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    total_data = [latent2_data, latent15_data, latent25_data]
    for i in range(len(latent_size)):
        for j in range(len(layers)):
            ax[i].plot(total_data[i][j][:600], label="{} LSTM Layer(s)".format(layers[j]))
        ax[i].set_title("Latent Size {}".format(latent_size[i]))
        ax[i].set_ylabel("Test Loss")
        ax[i].set_xlabel("Epoch")
        ax[i].set_yscale('log')
        ax[i].legend()
        ax[i].set_yscale('log')
    fig.suptitle("Testing Loss Comparison for Different Number of LSTM Layers")
    # plt.tight_layout()
    # plt.show()
    plt.savefig("./LSTM_SSE_NoKL_NoXY_layer_compare.png")
            

def plot_KL_schedule():
    epoch = np.arange(500)
    beta = 1/(1+np.exp(-0.08*(epoch-250)))
    plt.plot(epoch, beta) 
    plt.title("KL Cost Weight Schedule")
    plt.ylabel("Weight on KL Div")
    plt.xlabel("Epoch")
    plt.savefig("./KL_weight_schedule.png")

def plot_lstm_KL_schedule():
    test_loss = np.genfromtxt("./plot_data/lstm_anneal_KL/run-mj_state_lstm_FF_Relu_less_SSE_randInit_2layer32_AnnealedKL_NoXY_latent25_layer1_hidden64-tag-Test_Loss.csv", delimiter=",", skip_header=1, usecols=(2))
    train_loss = np.genfromtxt("./plot_data/lstm_anneal_KL/run-mj_state_lstm_FF_Relu_less_SSE_randInit_2layer32_AnnealedKL_NoXY_latent25_layer1_hidden64-tag-Train_Loss.csv", delimiter=",", skip_header=1, usecols=(2))
    KL_loss = np.genfromtxt("./plot_data/lstm_anneal_KL/run-mj_state_lstm_FF_Relu_less_SSE_randInit_2layer32_AnnealedKL_NoXY_latent25_layer1_hidden64-tag-Train_Loss_KL.csv", delimiter=",", skip_header=1, usecols=(2))
    SSE_loss = np.genfromtxt("./plot_data/lstm_anneal_KL/run-mj_state_lstm_FF_Relu_less_SSE_randInit_2layer32_AnnealedKL_NoXY_latent25_layer1_hidden64-tag-Train_Loss_SSE.csv", delimiter=",", skip_header=1, usecols=(2))
    num_epoch = len(KL_loss)
    # print(num_epoch)

    epoch = np.arange(num_epoch)
    beta = 1/(1+np.exp(-0.08*(epoch-250)))
    weighted_KL_loss = np.multiply(KL_loss, beta)
    full_train_loss = weighted_KL_loss + SSE_loss
    rand_inds = np.divide(full_train_loss, train_loss)
    # print(train_loss - (weighted_KL_loss+SSE_loss))
    # print(rand_inds)
    # exit()
    zeros = np.zeros(num_epoch)

    fig, ax = plt.subplots(2, 2, figsize=(16, 6), constrained_layout=True)
    ax[0][0].plot(train_loss, label="Training Loss", color='dodgerblue')
    ax[0][0].plot(weighted_KL_loss, color='darkviolet')
    ax[0][0].fill_between(epoch, weighted_KL_loss, label="KL Loss", color='darkviolet', alpha=0.5)
    ax[0][0].fill_between(epoch, train_loss, weighted_KL_loss, label="SSE Loss", color='orange', alpha=0.5)
    ax[0][0].set_ylim(1e-4, 400)
    ax[0][0].legend()
    ax[0][0].set_yscale('log')
    ax[0][0].set_title("Training Loss Components")
    ax[0][0].set_ylabel("Training Loss")
    ax[0][0].set_xlabel("Epoch")

    ax[0][1].plot(SSE_loss)
    ax[0][1].set_yscale('log')
    ax[0][1].set_title("SSE Loss")
    ax[0][1].set_ylabel("SSE Loss")
    ax[0][1].set_xlabel("Epoch")

    ax[1][1].plot(KL_loss)
    ax[1][1].set_yscale('log')
    ax[1][1].set_title("KL Divergence Loss")
    ax[1][1].set_ylabel("KL Loss")
    ax[1][1].set_xlabel("Epoch")

    ax[1][0].plot(beta)
    ax[1][0].set_title("KL Loss Weight Schedule")
    ax[1][0].set_ylabel("KL Loss Weighting")
    ax[1][0].set_xlabel("Epoch")

    fig.suptitle("Training Loss Comparison for RNN VAE with KL Loss Schedule")
    # plt.show()
    plt.savefig("./lstm_KL_weight_schedule.png")

# plot_KL_schedule()
# plot_LSTM_layer_sweep()
plot_lstm_KL_schedule()