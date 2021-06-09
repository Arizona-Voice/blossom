# -*- coding: utf-8 -*-

import os
import torch
import librosa
import numpy as np


def str2bool(value):
    return str(value).lower() in ('yes', 'true', 't', '1')

def get_from_registry(key, registry):
    if hasattr(key, 'lower'):
        key = key.lower()

    if key in registry:
        return registry[key]
    else:
        raise ValueError(f"Key `{key}` not supported, available options: {registry.keys()}")

def transform(path, sr=16000):
    sig, sr = librosa.load(path, sr)
    spec = librosa.feature.mfcc(sig, sr=sr, n_mfcc=40)
    x = np.array(spec, np.float32, copy=False)
    x = torch.from_numpy(x)
    
    return x
