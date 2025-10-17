import torch_audiomentations
import os
import torchaudio
import torch
import torch.functional as F
import random
from torch import Tensor, nn
from torchaudio.transforms import Resample
from pathlib import Path
from tqdm import tqdm

class Gain(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._aug = torch_audiomentations.Gain(*args, **kwargs)

    def __call__(self, wav: Tensor):
        x = wav.unsqueeze(1)
        return self._aug(x).squeeze(1)

class RandomNoise:
    def __init__(self, root, sr=16000, snr_range=(20, 30), prob=1.0):
        super().__init__()
        self.prob = prob
        self.snr_range = snr_range
        self.root = root
        self.sr = sr
        self.noise_filenames = sorted(str(p) for p in Path(root).rglob("*.wav"))
        if not self.noise_filenames:
            raise FileNotFoundError(f"No .wav files found under '{root}'")
        self.resampler_cache = {}

    def _load_noise(self, target_length, target_sr):
        noise_path = random.choice(self.noise_filenames)
        noise_waveform, noise_sr = torchaudio.load(noise_path)

        if noise_sr != target_sr:
            if noise_sr not in self.resampler_cache:
                self.resampler_cache[noise_sr] = Resample(noise_sr, target_sr, resampling_method="sinc_interp_kaiser")
            noise_waveform = self.resampler_cache[noise_sr](noise_waveform)

        repeat_factor = (target_length // noise_waveform.shape[1]) + 1
        noise_waveform = noise_waveform.repeat(1, repeat_factor)
        noise_waveform = noise_waveform[:, :target_length]

        noise_waveform = noise_waveform.nan_to_num(0)
        return noise_waveform

    def __call__(self, wav: Tensor):
        if random.random() > self.prob:
            return wav

        try:
            target_length = wav.shape[1]
            noise = self._load_noise(target_length, self.sr)
            snr = torch.tensor([random.uniform(*self.snr_range)])
            noisy_wav = torchaudio.functional.add_noise(wav, noise, snr)
        except Exception:
            return wav
        
        return noisy_wav
    
class RandomImpulseResponse:
    def __init__(self, root, sr=16000, prob=0.8):
        super().__init__()
        self.prob = prob
        self.root = root
        self.ir_filenames = sorted(str(p) for p in Path(root).rglob("*.wav"))
        if not self.ir_filenames:
            raise FileNotFoundError(f"No .wav files found under '{root}'")
        self.sr = sr
        self.resampler_cache = {}

    def _load_ir(self, target_sr: Tensor):
        ir_path = random.choice(self.ir_filenames)
        ir_waveform, sr_ir = torchaudio.load(ir_path)

        if sr_ir != target_sr:
            if sr_ir not in self.resampler_cache:
                self.resampler_cache[sr_ir] = Resample(sr_ir, target_sr, resampling_method="sinc_interp_kaiser")
            ir_waveform = self.resampler_cache[sr_ir](ir_waveform)

        ir_waveform = ir_waveform[...,int(ir_waveform.abs().argmax().item()):]
        ir_waveform /= (ir_waveform[...,0].item() + 1e-8)

        return ir_waveform
    
    def __call__(self, wav: torch.Tensor):
        if random.random() > self.prob:
            return wav

        try:
            ir_waveform = self._load_ir(self.sr)
            ir_waveform = ir_waveform.unsqueeze(0)
            if wav.dim() == 2:
                wav = wav.unsqueeze(0)

            processed_wav = torchaudio.functional.fftconvolve(wav, ir_waveform, mode='full')
            processed_wav = processed_wav[...,:wav.shape[-1]]
        except Exception:
            return wav
        
        processed_wav = processed_wav.nan_to_num(0).squeeze(0)
        return processed_wav
