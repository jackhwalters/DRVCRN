import torch
import torchaudio.functional as taf


def drvcrn_train(drvcrn, noisy_names, clean_names, optimizer,\
        train_noisy_waveform, train_clean_waveform, device, hop_length, win_length, criterion):
    assert len(noisy_names) == len(clean_names)
    running_loss = 0
    for i in range(len(noisy_names)):
        optimizer.zero_grad()

        noisy = train_noisy_waveform[i].to(device)
        clean = train_clean_waveform[i].to(device)

        noisy_STFT = torch.stft(noisy, n_fft=512, hop_length=hop_length, win_length=win_length)

        noisy_mag, noisy_phase = taf.magphase(noisy_STFT)

        mask = drvcrn(noisy_mag)
        predict = noisy_mag * mask

        output_STFT = torch.cat([
            (predict * torch.cos(noisy_phase)).unsqueeze(-1),
            (predict * torch.sin(noisy_phase)).unsqueeze(-1)
        ], dim=-1)
        output_waveform = torch.istft(output_STFT, n_fft=512, hop_length=hop_length,
                                      win_length=win_length, length = noisy.shape[0])

        loss = -criterion(output_waveform, clean)
        loss.backward()
        optimizer.step()
        print('loss.item(): ', loss.item())
        running_loss += loss.item()

    return -running_loss / len(noisy_names)


def drvcrn_test(drvcrn, noisy_names, clean_names, criterion):
    assert len(noisy_names) == len(clean_names)
    running_loss = 0
    for i in range(len(noisy_names)):

        noisy = test_noisy_waveform[i].to(device)
        clean = test_clean_waveform[i].to(device)

        noisy_STFT = torch.stft(noisy, n_fft=512, hop_length=hop_length, win_length=win_length)

        noisy_mag, noisy_phase = taf.magphase(noisy_STFT)

        mask = drvcrn(noisy_mag)
        predict = noisy_mag * mask

        output_STFT = torch.cat([
            (predict * torch.cos(noisy_phase)).unsqueeze(-1),
            (predict * torch.sin(noisy_phase)).unsqueeze(-1)
        ], dim=-1)
        output_waveform = torch.istft(output_STFT, n_fft=512, hop_length=hop_length,
                                      win_length=win_length, length = noisy.shape[0])

        loss = -criterion(output_waveform, clean)

        running_loss += loss.item()

    return -running_loss / len(noisy_names)
