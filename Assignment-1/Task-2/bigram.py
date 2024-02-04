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
        print("Model Initialized 🟢")

    def set_laplace_matrix(self):
        row_sums = self.matrix.sum(axis=1)
        self.laplace_matrix = (self.matrix + 1) / (
            row_sums[:,np.newaxis] + len(self.token)
        )
        return

    def get_kn_matrix(self, d = 0.75):    
        bigram_matrix = self.matrix
        unigram_count = bigram_matrix.sum(axis=1)
        
        unique_bigrams = np.where(bigram_matrix != 0, 1, 0).sum()
        
        alpha = []
        for row in range(len(bigram_matrix)):
            alpha.append(d * np.where(bigram_matrix[row] !=0,1,0).sum()/unigram_count[row])
        
        context_count = []
        for col in range(len(bigram_matrix[0])):
            context_count.append(np.where(bigram_matrix[:,col] !=0,1,0).sum())
        
            
        self.kn_matrix = np.zeros((len(self.token), len(self.token)), dtype=float)
        
        for row in tqdm(range(len(bigram_matrix)), desc="Calculating Kneser-Ney Matrix..."): 
            for col in range(len(bigram_matrix[row])):
                continuation = context_count[col]/ unique_bigrams
                self.kn_matrix[row][col] = max(bigram_matrix[row][col] - d, 0) / unigram_count[row]
                self.kn_matrix[row][col] += alpha[row] * continuation
                
        return self.kn_matrix

                    
        
    def get_normal_matrix(self):
        return self.normalized_matrix

    def get_count_matrix(self):
        return self.matrix

    def get_laplace_matrix(self):
        return self.laplace_matrix

    def get_token(self):
        return self.ordered_tokens

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

        self.ordered_tokens = sorted(list(self.token))

        for i in range(len(self.ordered_tokens)):
            self.token_to_indice[self.ordered_tokens[i]] = i
        print("Tokens Set 🟢")
        return

    def find_indice(self, token):
        return self.token_to_indice[token]

    def calculate_bigrams(self):
        no_of_tokens = len(self.token)
        len_of_corpus = len(self.split_corpus)
        self.matrix = np.zeros((no_of_tokens, no_of_tokens), dtype=float)
        for i in tqdm(range(len_of_corpus - 1), desc="Populating Bigram Matrix..."):
            x = self.find_indice(self.split_corpus[i])
            x_plus_1 = self.find_indice(self.split_corpus[i + 1])
            self.matrix[x, x_plus_1] += 1

        row_sums = self.matrix.sum(axis=1)

        # Divide each element in a row by the sum of that row
        self.normalized_matrix = self.matrix / row_sums[:, np.newaxis]
        self.set_laplace_matrix()

        print("All Matrices Calculated 🟢")
