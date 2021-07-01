from tqdm import tqdm
from collections import Counter
import logging 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
import numpy as np
logger = logging.getLogger(__name__)

class W2VCorpus:
    def __init__(
        self,
        path: str,
        voc_mac_size: int=40000,
        min_word_freq: int=20,
        max_corpus_size: int=5e6
    ) -> None:
        corpus = []
        sentences = []
        with open(path, 'r') as f:
            for text in f:
                corpus.append(text.split())
                sentences.append(text)
        corpus = np.array(corpus)
        self.corpus = corpus
        most_freq_word = \
            Counter(' '.join(sentences).split()).most_common(
                voc_mac_size)
        most_freq_word = np.array(most_freq_word)
        most_freq_word = \
            most_freq_word[most_freq_word[:, 1].astype(int) \
                > min_word_freq]
        logger.info(f"Vocabulary size: {len(most_freq_word)}")
        self.vocabulary = set(most_freq_word[:, 0])
        self.vocabulary.update(['<PAD>'])
        self.vocabulary.update(['<UNK>'])
        self.word_freq = most_freq_word
        self.idx_to_word = dict(list(enumerate(self.vocabulary)))
        self.word_to_idx = \
            dict([(i[1], i[0]) for i in enumerate(self.vocabulary)])
        self.W = None
        self.P = None
        self.positive_pairs = None

    def make_positive_dataset(self, window_size=2):
        """Produce positive examples for skip-gram or CBOW
        through corpus
        """
        logger.info("Creating positive dataset")
        if not self.W is None:
            return self.W, self.P
        W = []
        P = []
        pbar = tqdm(self.corpus)
        pbar.set_description("Creating context dataset")
        for message in pbar:
            if len(self.corpus) == 1:
                iter_ = tqdm(enumerate(message), \
                    total=len(message))
            else:
                iter_ = enumerate(message)

            for idx, word in iter_:
                if word not in self.vocabulary:
                    word = '<UNK>'
                start_idx = max(0, idx - window_size)
                end_idx = min(len(message), idx + window_size + 1)
                pos_in_window = window_size
                if idx - window_size < 0:
                    # The beginning of the sentences
                    pos_in_window += idx - window_size

                contex_words = message[start_idx:end_idx]
                # Delete context words
                contex_words = np.delete(contex_words, pos_in_window)
                filtered_contex_words = []

                for contex_word in contex_words:
                    if contex_word in self.vocabulary:
                        filtered_contex_words.append(contex_word)
                    else:
                        filtered_contex_words.append('<UNK>')
                while len(filtered_contex_words) < 2 * window_size:
                    # Fill in the dataset with <PAD> tokens to ensure
                    # equals input's size
                    filtered_contex_words.append('<PAD>')
                W.append(self.word_to_idx[word])
                contex_word_idx = [self.word_to_idx[contex_word] \
                    for contex_word in filtered_contex_words]
                P.append(contex_word_idx)
        self.W = W
        self.P = P
        del self.corpus
        return W, P

    def make_positive_pairs(self):
        logger.info("Creating positive pairs")
        if not self.positive_pairs is None:
            return self.positive_pairs
        if self.W is None:
            self.make_positive_dataset(window_size=2)
        pairs = []
        pbar = tqdm(zip(self.W, self.P), total=len(self.W))
        pbar.set_description("Creating positive pairs")
        for w, p in pbar:
            for now_p in p:
                if now_p != self.word_to_idx['<PAD>']:
                    pairs.append([w, now_p])
        self.positive_pairs = pairs
        return pairs
      
class W2VDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
    
    def __getitem__(self, idx):
        return {
            'word': torch.tensor(self.pairs[idx][0]),
            'context': torch.tensor(self.pairs[idx][1]),
        }
    def __len__(self):
        return len(self.pairs)
