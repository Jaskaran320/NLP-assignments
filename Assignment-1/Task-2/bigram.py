import re
from tqdm import tqdm
import numpy as np


class BigramLM:
    def __init__(self, corpus):
        self.corpus = re.sub("\n", " ", corpus)
        self.split_corpus = self.corpus.split()
        self.token = set()
        self.token_to_indice = {}
        self.normalized_matrix = None
        self.matrix = None
        self.laplace_matrix = None
        print("MODEL INITITATED")

    def set_laplace_matrix(self):
        row_sums = self.matrix.sum(axis=1)
        self.laplace_matrix = (self.matrix + 1) / (
            row_sums[:, np.newaxis] + len(self.token)
        )
        return

    def get_normal_matrix(self):
        return self.normalized_matrix

    def get_count_matrix(self):
        return self.matrix

    def get_laplace_matrix(self):
        return self.laplace_matrix

    def get_token(self):
        return self.ordered_list

    def get_corpus(self):
        return self.split_corpus

    def get_sum(self):
        return self.matrix.sum(axis=1)

    # May need this if we want to work for a worse corpus
    def check_remove_punctuation(self):
        pass

    def set_token(self):
        for i in self.split_corpus:
            if i not in self.token:
                self.token.add(i)

        self.ordered_list = list(self.token)

        for i in range(len(self.ordered_list)):
            self.token_to_indice[self.ordered_list[i]] = i
        print("Done")
        return

    def find_indice(self, token):
        return self.token_to_indice[token]

    def calculate_bigrams(self):        

        no_of_tokens = len(self.token)
        len_of_corpus = len(self.split_corpus)
        self.matrix = np.zeros((no_of_tokens, no_of_tokens), dtype=float)
        for i in tqdm(range(len_of_corpus - 1), desc="Processing"):
            x = self.find_indice(self.split_corpus[i])
            x_plus_1 = self.find_indice(self.split_corpus[i + 1])
            self.matrix[x, x_plus_1] += 1

        row_sums = self.matrix.sum(axis=1)

        # Divide each element in a row by the sum of that row
        self.normalized_matrix = self.matrix / row_sums[:, np.newaxis]
        self.set_laplace_matrix()

        print(" All Matrix Calculated")
