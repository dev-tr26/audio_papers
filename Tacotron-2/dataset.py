import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchaudio
from torch.utils.data import Dataset
import librosa 
from tokenizer import Tokenizer 

def load_wav(path_to_audio, sr=22050):
    audio, original_Sr = torchaudio.load(path_to_audio)
    
    if sr!= original_Sr:
        audio = torchaudio.functional.resample(audio, orig_freq=original_Sr,new_freq=sr)
    
    return audio.squeeze(0)



def amplitude_to_db(x, min_db=-100):
    # 20 * torch.log10(1e-5) = 20 * -5 = -100  min of deb always -100
    clip_val = 10 ** (min_db/20)
    return 20 * torch.log10(x, torch.clamp(min=min_db))


def db_to_amp(x):
    return 10**(x/20)


# values from non-nvidia implementation, in nvidia implementation trained directly on log vals scaled 
# here rescale b/w -4 to 4 
# make sure every single spectrogram time step that we are predicting 

def normalize(x, min_db=-100, max_abs_val=4):
    
    