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
                 word_break_char: str = ' '):
        """
        Parameters
        ----------
        num_iter: int
            Number of iterations to run.
        pad_token: str
            Pad token added at the end of word.
        unk_token: str
            Unknown token.
        word_break_char: str
            Character to break words on.
        """
        self.num_iter = num_iter
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.word_break_char = word_break_char
        self.tokens = None

    def _encode_word(self,
                     word: str) -> List[int]:
        """
        Encode a word into a list of tokens.
        From the start of the word, find the longest substring that is a token.
        If no token is found, return the unknown token. 
        If a token is found, return the token and recursively call this function
        on the rest of the word.
        """        
        if word == '':
            return []

        longest_substr_idx = -1
        for i in range(len(word)):
            if word[:i+1] in self.tokens:
                longest_substr_idx = i
        
        if longest_substr_idx == -1:
            return [self.unk_token]
        
        return [word[:longest_substr_idx+1]] \
                + self._encode_word(word[longest_substr_idx+1:])

    def encode(self, 
               text: str) -> List[int]:
        words = text.split(self.word_break_char)
        paded_words = [word + self.pad_token for word in words]
        tokenized_text = []
        for paded_word in paded_words:
            tokenized_word = self._encode_word(paded_word)
            tokenized_text.extend(tokenized_word)

        return tokenized_text

    def decode(self, 
               tokens: List[int]) -> str:
        return ''.join(tokens)\
                 .replace(self.pad_token, self.word_break_char)

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

        for _ in tqdm.tqdm(range(self.num_iter)):
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
        for idx, (tokenized_word, _) in enumerate(tokenized_word_freq_tuples):
            tokenized_word_freq_tuples[idx][0] = re.sub(
                pattern = rf"(^|{self.word_break_char})" +
                          rf"({re.escape(byte_pair[0])})"+
                          rf"({self.word_break_char})"+
                          rf"({re.escape(byte_pair[1])})"+
                          rf"({self.word_break_char}|$)", 
                repl = r"\1\2\4\5", 
                string = tokenized_word)


