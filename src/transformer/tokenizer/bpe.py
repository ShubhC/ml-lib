from tokenizer import Tokenizer
from typing import List, Tuple
from collections import defaultdict
import re
import tqdm

class BypePairEncoding(Tokenizer):
    def __init__(self,
                 num_iter: int,
                 pad_token: str,
                 unk_token: str,
                 word_break_char: str = ''):
        self.num_iter = num_iter
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.word_break_char = word_break_char
        self.tokens = None

    def encode(self, 
               text: str) -> List[int]:
        pass

    def decode(self, 
               tokens: List[int]) -> str:
        pass

    def learn(self, 
              texts: List[str]):

        word_freq = defaultdict(int)
        for text in texts:
            # Get all words in text
            text = text.strip()
            words_in_text = text.split(self.word_break_char)

            # Prepare word frequency dictionary
            for word in words_in_text:
                word_freq[word] += 1

        # Create tokenized word freq tuples
        tokenized_word_freq_tuples = list()
        for word, freq in word_freq.items():
            tokenized_word = self.word_break_char.join(word) + self.pad_token
            tokenized_word_freq_tuples.append([tokenized_word, freq])

        for i in tqdm.tqdm(range(self.num_iter)):
            byte_pair_with_max_freq = self._get_byte_pair_with_max_freq(tokenized_word_freq_tuples)
            self._combine_byte_pair_in_all_words(tokenized_word_freq_tuples, 
                                                 byte_pair_with_max_freq)
            
        self.tokens = set()
        for tokenized_word, _ in tokenized_word_freq_tuples:
            self.tokens.update(tokenized_word.split(self.word_break_char))
        return list(self.tokens)
    
    def _get_byte_pair_with_max_freq(self,
                                     word_freq_tuples: List[List]) -> Tuple[str, str]:
        max_freq = -1
        max_freq_pair = None

        byte_pair_freq = defaultdict(int)
        for (word, freq) in word_freq_tuples:
            tokens = word.split(self.word_break_char)
            for i in range(len(tokens) - 1):
                byte_pair = tokens[i] + tokens[i + 1]
                byte_pair_freq[byte_pair] += freq

                if byte_pair_freq[byte_pair] > max_freq:
                    max_freq = byte_pair_freq[byte_pair]
                    max_freq_pair = [tokens[i], tokens[i + 1]]

        return max_freq_pair

    def _combine_byte_pair_in_all_words(self,
                                        tokenized_word_freq_tuples: List[List],
                                        byte_pair: Tuple[str, str]):
        pattern = re.compile(fr"\b({byte_pair[0]})\s({byte_pair[1]})\b")
        for idx, (tokenized_word, freq) in enumerate(tokenized_word_freq_tuples):
            tokenized_word_freq_tuples[idx][0] = pattern.sub(r"\1\2", tokenized_word)

