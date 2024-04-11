from abc import ABC, abstractmethod
from typing import List
import pickle
from pathlib import Path

class Tokenizer(ABC):

    num_iter: int
    pad_token: str
    unk_token: str

    @abstractmethod
    def encode(self, text: str) -> List[str]:
        pass

    @abstractmethod
    def decode(self, tokens: List[str]) -> str:
        pass

    @abstractmethod
    def learn(self, texts: List[str]):
        pass

    def save(self, path: str):
        pickle.dump(self, open(path, 'wb'))

    def load(self, path: str):
        self = pickle.load(open(path, 'rb'))