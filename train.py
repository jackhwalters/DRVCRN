from wav_loader import *
from network import *
from network_train_test import *
from si_snr import *
import matplotlib.pyplot as plt
import torch


dns_path = "/Users/jack/BigBois/dns_datas"
logs_path = "logs"

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = torch.device("cpu")

lr = 0.001
sr = 16000
n_epochs = 200

drvcrn = DRVCRN().to(device)
drvcrn.apply(weights_init)
criterion = SiSnr()
optimizer = torch.optim.Adam(drvcrn.parameters(), lr=lr)

win_length = int(25 * (sr / 1000))
hop_length = int(win_length / 4)
print(win_length)
print(hop_length)

train_test = get_train_test_name(dns_path)
train_noisy_names, train_clean_names, test_noisy_names, test_clean_names = \
    get_all_names(train_test, dns_path=dns_path)

train_noisy_waveform = []
train_clean_waveform = []
test_noisy_waveform = []
test_clean_waveform = []

train_noisy_waveform, train_clean_waveform, test_noisy_waveform, test_clean_waveform = \
    load_waveform(train_noisy_names, train_clean_names, test_noisy_names, test_clean_names, sr)


train_losses = []
test_losses = []
for i in range(n_epochs):
    train_epoch_loss = 0
    test_epoch_loss = 0

    train_epoch_loss = drvcrn_train(drvcrn, train_noisy_names, train_clean_names, optimizer, \
        train_noisy_waveform, train_clean_waveform, device, hop_length, win_length, criterion)
    train_losses.append(train_epoch_loss)

    test_epoch_loss = drvcrn_test(drvcrn, test_noisy_names, test_clean_names, criterion)
    test_losses.append(test_epoch_loss)

    print("Epoch {}: Train SI-SNR: {}, Test SI-SNR: {}".format(i+1, train_losses[-1], test_losses[-1]))
