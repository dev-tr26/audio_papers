import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchaudio
from torch.utils.data import Dataset
import librosa 
from tokenizer import Tokenizer 
import numpy as np

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
    x = (x - min_db) / min_db 
    x = 2 * max_abs_val * x - max_abs_val 
    x = torch.clip(x, min=-max_abs_val, max=max_abs_val)
    return x 


def denormalize(x, min_db=-100,max_abs_val=4):
    x = torch.clip(x, min=-max_abs_val, max=max_abs_val)
    x = (x + max_abs_val) / ( 2 *max_abs_val)
    x = x * -min_db + min_db
    return x



class AudioMelConversions:
    def __init__(self, n_mels=80, sr=22050, n_fft=1024, window_size=1024, hop_size=256, fmin=0, fmax=8000, center=False, min_db=-100, max_Scaled_abs=4):
        self.n_mels = n_mels
        self.sr = sr 
        self.n_fft = n_fft 
        self.window_size = window_size
        self.hop_size = hop_size 
        self.fmin = fmin 
        self.fmax = fmax 
        self.center = center 
        self.min_db = min_db
        self.max_Scaled_abs = max_Scaled_abs
        
        self.spec2mel = self._get_spec2mel_proj()
        self.mel2spec = torch.linalg.pinv(self.spec2mel)
        
    
    def _get_spec2mel_proj(self):
        mel = librosa.filters.mel(sr=self.sr, n_fft=self.n_fft, n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax)
        return torch.from_numpy(mel)
    
    
    def audio2mel(self, audio, is_norm = False):
        
        if not isinstance(audio, torch.Tensor):
            audio = torch.tensor(audio, dtype=torch.float32)
    
        spectrogram = torch.stft(input=audio, nfft=self.n_fft, win_length=self.window_size, hop_length=self.hop_size, center=self.center,pad_mode="reflect" ,onesided=True , return_complex=True)
        spectrogram = torch.abs(spectrogram)
        
        mel = torch.matmul(self.spec2mel.to(spectrogram.device), spectrogram)
        
        mel = amplitude_to_db(mel, self.min_db)
        
        if is_norm:
            mel = normalize(mel, min_db=self.min_db, max_abs_val=self.max_Scaled_abs)
        
        return mel 
    
    

    def mel2audio(self, mel, is_denorm=False, griffin_lim_iters=60):
        if is_denorm:
            mel = denormalize(mel, min_db=self.min_db, max_abs_val=self.max_Scaled_abs)
            
        mel = db_to_amp(mel)
        spec = torch.matmul(self.mel2spec.to(mel.device), mel).cpu().numpy()
        
        audio = librosa.griffinlim(s=spec, n_iter=griffin_lim_iters, hop_length=self.hop_size, win_length=self.window_size, n_fft=self.n_fft,window="hann")
        
        audio *= 32767 / max(0.01, np.max(np.abs(audio)))
        audio = audio.astype(np.int16)
        return audio 
    
    
    
    
def build_padding_mask(lengths):
        
    B = lengths.size(0)
    T = torch.max(lengths).item()
        
    mask = torch.zeros(B, T)
    for i in range(B):
        mask[i, lengths[i]:] = 1 
            
    return mask.bool()
    
    
    

class TTSDataset(Dataset):
    
    def __init__(self, metadata_path, sr=22050, n_fft=1024, window_size=1024, hop_size=256, fmin=0, fmax=8000, n_mels=80, center=False, normalized=False, min_db=-100, max_scaled_abs=4):
        
        self.metadata = pd.read_csv(metadata_path)  
        self.sr  = sr
        self.n_fft = n_fft 
        self.window_size = window_size
        self.hop_size = hop_size
        self.fmin = fmin
        self.fmax = fmax
        self.n_mels = n_mels
        self.center = center
        self.normalized = normalized
        self.min_db = min_db
        self.max_scaled_abs = max_scaled_abs
        
        self.trancript_length = [len(Tokenizer().encode(t)) for t in self.metadata["normalized_transcript"]]
        
        self.audio_proessor = AudioMelConversions(n_mels=self.n_mels,sr=self.sr, n_fft=self.n_fft, window_size=self.window_size, hop_size=self.hop_size, fmin=self.fmin, fmax=self.fmax, center=self.center, min_db = self.min_db, max_scaled_abs=self.max_scaled_abs ) 
    
    
    def __len__(self):
        return len(self.metadata)
    
    
    def __getitem__(self, index):
        
        sample = self.metadata.iloc[index]
        audio_path = sample["file_path"]
        transcript = sample["normalized_transcript"]
        
        audio = load_wav(audio_path, sr=self.sr)
        
        mel = self.audio_processor.audio2mel(audio, is_norm=True)
        
        return transcript, mel.squeeze(0)
    
    
    
    def TTSCollator():
        
        tokenizer = Tokenizer()
        
        def _collate_fn(batch):
            texts = [tokenizer.encode(b[0]) for b in batch]
            mels = [b[1] for b in batch]
            
            #lens of txt and mels 
            input_len = torch.tensor([t.shape[0] for t in texts], dtype=torch.long)
            output_len = torch.tensor([m.shape[1] for m in mels], dtype=torch.long)
            
            # sort by txt lens as it will be used for packed tensors 
            input_lens , sorted_index = input_lens.sort(descending=True)
            texts = [texts[i] for i  in sorted_index]
            mels = [mels[i] for i in sorted_index]
            output_lens = output_lens[sorted_index]
            
            # padding txts 
            
            padded_texts = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True)
            
            # pad mel sequences 
            max_output_len = output_lens[0].item()
            num_mels = mels[0].shape[0]
            
            mel_padded = torch.zeros((len(mels), num_mels, max_output_len))
            gate_padded = torch.zeros((len(mels), max_output_len))
            
            for i, mel in enumerate(mels):
                t = mel.shape[1]
                mel_padded[i, :, :t] = mel 
                gate_padded[i, t-1:] = 1
            
            mel_padded = mel_padded.transpose(1,2)

            return padded_texts , input_lens, mel_padded, gate_padded, build_padding_mask(input_len), build_padding_mask(output_len)

        return _collate_fn