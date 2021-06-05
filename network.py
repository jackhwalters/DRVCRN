from torch import nn
import torch


def weights_init(model):
    if isinstance(model, nn.Linear):
        torch.nn.init.xavier_uniform_(model.weight)

class DRVCRN(nn.Module):
    def __init__(self):
        super(DRVCRN, self).__init__()
        self.conv_encode1 = nn.Conv2d(1, 32, kernel_size=(5,2), stride=(2,1), padding=(2,0))
        self.bne1 = nn.BatchNorm2d(32)
        self.pre1 = nn.PReLU()
        self.conv_encode2 = nn.Conv2d(32, 64, kernel_size=(5,2), stride=(2,1), padding=(2,0))
        self.bne2 = nn.BatchNorm2d(64)
        self.pre2 = nn.PReLU()
        self.conv_encode3 = nn.Conv2d(64, 128, kernel_size=(5,2), stride=(2,1), padding=(2,0))
        self.bne3 = nn.BatchNorm2d(128)
        self.pre3 = nn.PReLU()
        self.conv_encode4 = nn.Conv2d(128, 256, kernel_size=(5,2), stride=(2,1), padding=(2,0))
        self.bne4 = nn.BatchNorm2d(256)
        self.pre4 = nn.PReLU()
        self.conv_encode5 = nn.Conv2d(256, 256, kernel_size=(5,2), stride=(2,1), padding=(2,0))
        self.bne5 = nn.BatchNorm2d(256)
        self.pre5 = nn.PReLU()
        self.conv_encode6 = nn.Conv2d(256, 256, kernel_size=(5,2), stride=(2,1), padding=(2,0))
        self.bne6 = nn.BatchNorm2d(256)
        self.pre6 = nn.PReLU()

        self.lstm = nn.LSTM(input_size=1280, hidden_size=128, num_layers=4, bidirectional=False)
        self.fc = nn.Linear(128, 1280)

        self.conv_decode1 = nn.ConvTranspose2d(256, 256, kernel_size=(5,2), stride=(2,1), padding=(2,0))
        self.bnd1 = nn.BatchNorm2d(256)
        self.prd1 = nn.PReLU()
        self.conv_decode2 = nn.ConvTranspose2d(256, 256, kernel_size=(5,2), stride=(2,1), padding=(2,0))
        self.bnd2 = nn.BatchNorm2d(256)
        self.prd2 = nn.PReLU()
        self.conv_decode3 = nn.ConvTranspose2d(256, 128, kernel_size=(5,2), stride=(2,1), padding=(2,0))
        self.bnd3 = nn.BatchNorm2d(128)
        self.prd3 = nn.PReLU()
        self.conv_decode4 = nn.ConvTranspose2d(128, 64, kernel_size=(5,2), stride=(2,1), padding=(2,0))
        self.bnd4 = nn.BatchNorm2d(64)
        self.prd4 = nn.PReLU()
        self.conv_decode5 = nn.ConvTranspose2d(64, 32, kernel_size=(5,2), stride=(2,1), padding=(2,0))
        self.bnd5 = nn.BatchNorm2d(32)
        self.prd5 = nn.PReLU()
        self.conv_decode6 = nn.ConvTranspose2d(32, 1, kernel_size=(5,2), stride=(2,1), padding=(2,0))

    def forward(self, x):
        net_in = x.view(-1, 1, x.shape[0], x.shape[1])
        e1 = self.pre1(self.bne1(self.conv_encode1(net_in)))
        e2 = self.pre2(self.bne2(self.conv_encode2(e1)))
        e3 = self.pre3(self.bne3(self.conv_encode3(e2)))
        e4 = self.pre4(self.bne4(self.conv_encode4(e3)))
        e5 = self.pre5(self.bne5(self.conv_encode5(e4)))
        e6 = self.pre6(self.bne6(self.conv_encode6(e5)))

        latent_shape = e6.shape
        reshaped1 = e6.reshape(latent_shape[0], latent_shape[1] * latent_shape[2], latent_shape[3]).permute(0, 2, 1)
        lstm_out, _ = self.lstm(reshaped1)
        fx_out = self.fc(lstm_out)
        reshaped2 = fx_out.permute(0, 2, 1).reshape(latent_shape[0], latent_shape[1], latent_shape[2], latent_shape[3])

        d1 = self.prd1(self.bnd1(self.conv_decode1(reshaped2 + e6)))
        d2 = self.prd2(self.bnd2(self.conv_decode2(d1 + e5)))
        d3 = self.prd3(self.bnd3(self.conv_decode3(d2 + e4)))
        d4 = self.prd4(self.bnd4(self.conv_decode4(d3 + e3)))
        d5 = self.prd5(self.bnd5(self.conv_decode5(d4 + e2)))
        d6 = self.conv_decode6(d5 + e1)

        net_out = torch.squeeze(d6)

        return net_out
