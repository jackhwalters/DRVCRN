## Deep Real Valued Convolutional Recurrent Network

This is a real-valued implementation of [this paper](https://arxiv.org/abs/2008.00264) for speech enhancement

1. Change the "dns_path" in train.py to the directory of the [Microsoft Scalable Noisy Speech Dataset](https://github.com/microsoft/MS-SNSD)
```
dns-datas/
   -clean/
   -noise/
   -noisy/
```

2. To train, run
```
python train.py
```
