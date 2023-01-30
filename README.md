# 読み上げAPI
[Tacotron2](https://github.com/NVIDIA/tacotron2)を利用した読み上げAPI。

## 動作確認済み環境
GPU:NVIDIA RTX 3060

OS:Ubuntu 20.04.5 LTS

Python:3.7.1

CUDA:11.7

## Setup
1. Download and extract the [LJ Speech dataset](https://keithito.com/LJ-Speech-Dataset/)
2. Clone this repo: `git clone https://github.com/NVIDIA/tacotron2.git`
3. CD into this repo: `cd tacotron2`
4. Initialize submodule: `git submodule init; git submodule update`
5. Install [PyTorch 1.0]
6. Install [Apex](https://github.com/nvidia/apex)
7. Install python requirements: `pip install -r requirements.txt`

## APIサーバーの起動
api.pyを編集します。
```:python
...

#チェックポイントの読み込み
checkpoint_path = "" ##チェックポイントのパスに変更 
model = load_model(hparams)
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
_ = model.cuda().eval()

...

```
APIサーバーを起動します。
```:bash
python api.py
```


