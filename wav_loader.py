import os
import librosa
import torch


def load_wav(path, frame_dur, sr):
    signal, _ = lib.load(path, sr=sr)
    win = int(frame_dur / 1000 * sr)
    return torch.tensor(np.split(signal, int(len(signal) / win), axis=0))

def get_train_test_name(dns_path):
    all_name = []
    for i in os.walk(os.path.join(dns_path, "noisy")):
        for name in i[2]:
            all_name.append(name)
    train_names = all_name[:-len(all_name) // 5]
    test_names = all_name[-len(all_name) // 5:]
    print("number of train items: ",len(train_names))
    print("number of test items: ",len(test_names))
    data = {"train": train_names, "test": test_names}
    return data

def get_all_names(train_test, dns_path):
    train_names = train_test["train"]
    test_names = train_test["test"]

    train_noisy_names = []
    train_clean_names = []
    test_noisy_names = []
    test_clean_names = []
    gender_names = []

    for name in train_names:
        code = str(name).split('_')[-1]
        clean_file = os.path.join(dns_path, 'clean', code)
        noisy_file = os.path.join(dns_path, 'noisy', name)
        train_clean_names.append(clean_file)
        train_noisy_names.append(noisy_file)
    for name in test_names:
        code = str(name).split('_')[-1]
        clean_file = os.path.join(dns_path, 'clean', code)
        noisy_file = os.path.join(dns_path, 'noisy', name)
        test_clean_names.append(clean_file)
        test_noisy_names.append(noisy_file)
    return train_noisy_names, train_clean_names, test_noisy_names, test_clean_names

def load_waveform(train_noisy_names, train_clean_names, test_noisy_names, test_clean_names, sr):
    train_noisy_waveform = []
    train_clean_waveform = []
    test_noisy_waveform = []
    test_clean_waveform = []

    assert len(train_noisy_names) == len(train_clean_names)
    assert len(test_noisy_names) == len(test_clean_names)

    for i in range(len(train_noisy_names)):
        audio_noisy, _ = librosa.load(str(train_noisy_names[i]), sr)
        audio_noisy = torch.from_numpy(audio_noisy)
        audio_clean, _ = librosa.load(str(train_clean_names[i]), sr)
        audio_clean = torch.from_numpy(audio_clean)

        train_noisy_waveform.append(audio_noisy)
        train_clean_waveform.append(audio_clean)


    for i in range(len(test_noisy_names)):
        audio_noisy, _ = librosa.load(str(test_noisy_names[i]), sr)
        audio_noisy = torch.from_numpy(audio_noisy)
        audio_clean, _ = librosa.load(str(test_clean_names[i]), sr)
        audio_clean = torch.from_numpy(audio_clean)

        test_noisy_waveform.append(audio_noisy)
        test_clean_waveform.append(audio_clean)

    return train_noisy_waveform, train_clean_waveform, test_noisy_waveform, test_clean_waveform
