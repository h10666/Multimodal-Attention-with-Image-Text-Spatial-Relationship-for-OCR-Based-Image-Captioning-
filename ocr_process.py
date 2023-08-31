import torch
import numpy as np
from fasttext import load_model
from cphoc import build_phoc as _build_phoc_raw

_alphabet = {"a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z","0","1","2","3","4","5","6","7","8","9"}  # NoQA

def build_phoc(token):
    token = token.lower().strip()
    token = ''.join([c for c in token if c in _alphabet])
    phoc = _build_phoc_raw(token)
    phoc = np.array(phoc, dtype=np.float32)
    return phoc

class PhocProcessor():
    """
    Compute PHOC features from text tokens
    """
    def __init__(self):
        self.PAD_INDEX = 0

    def map_ocr(self, tokens):

        phoc_dim = 604
        output = torch.full(
            (len(tokens), phoc_dim),
            fill_value=self.PAD_INDEX,
            dtype=torch.float,
        )

        for idx, token in enumerate(tokens):
            output[idx] = torch.from_numpy(build_phoc(token))

        return output

class WordToVectorDict:
    def __init__(self, model):
        self.model = model

    def __getitem__(self, word):
        # Check if mean for word split needs to be done here
        return np.mean([self.model.get_word_vector(w) for w in word.split(" ")], axis=0)

class FastTextProcessor():
    def __init__(self, model_file):
        self.model = load_model(model_file)
        # String to Vector
        self.stov = WordToVectorDict(self.model)
        self.PAD_INDEX = 0

    def map_ocr(self, tokens):

        output = torch.full(
            (len(tokens), self.model.get_dimension()),
            fill_value=self.PAD_INDEX,
            dtype=torch.float,
        )

        for idx, token in enumerate(tokens):
            output[idx] = torch.from_numpy(self.stov[token])

        return output

