#依存ライブラリのインポート
import matplotlib
import matplotlib.pylab as plt

import IPython.display as ipd

import sys
sys.path.append('waveglow/')
import numpy as np
import torch

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
from denoiser import Denoiser
import pyopenjtalk
import soundfile as sf

#各種設定
def plot_data(data, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='bottom', 
                       interpolation='none')

hparams = create_hparams()
hparams.sampling_rate = 22050

#チェックポイントの読み込み
checkpoint_path = "models/ringo/checkpoint_29000"
model = load_model(hparams)
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
_ = model.cuda().eval()

#waveglow学習済みモデルの読み込み
waveglow_path = 'waveglow_256channels_universal_v5.pt'
waveglow = torch.load(waveglow_path)['model']
waveglow.cuda().eval()
for k in waveglow.convinv:
    k.float()
denoiser = Denoiser(waveglow)


#テキストの入力
def generate(input:str):
    text = pyopenjtalk.g2p(input, kana=False).replace('pau',',').replace(' ','') + '.' + "\n"
    sequence = np.array(text_to_sequence(text, ['basic_cleaners']))[None, :]
    sequence = torch.autograd.Variable(
        torch.from_numpy(sequence)).cuda().long()

    #メルスペクトログラムの生成
    mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
    plot_data((mel_outputs.float().data.cpu().numpy()[0],
            mel_outputs_postnet.float().data.cpu().numpy()[0],
            alignments.float().data.cpu().numpy()[0].T))

    #音声ファイルの生成
    audio = waveglow.infer(mel_outputs_postnet, sigma=0.666);
    sf.write('test.wav', audio[0].data.cpu().numpy().astype(np.float32), hparams.sampling_rate)
    print('正常に音声が生成されました。')


try:
    input = input("文字を入力:")
    generate(input)
except:   
    print('error.')
