import re
from tqdm import tqdm
import numpy as np
class BigramLM:
    def __init__(self,corpus):
        self.corpus=re.sub("\n"," ",corpus)
        self.token=set()
        self.token_to_indice={}
        print("MODEL INITITATED b")
        
    # May need this if we want to work for a worse corpus
    def check_remove_punctuation():
        pass

    def get_token(self):
        self.split_corpus=self.corpus.split()
        for i in self.split_corpus:
            if i not in self.token:
                self.token.add(i)
        
        self.ordered_list=list(self.token)

        for i in range(len(self.ordered_list)):
            self.token_to_indice[self.ordered_list[i]]=i
        print("Done")
        return 
    
    def find_indice(self,token):
        return self.token_to_indice[token]
    
    def get_matrix(self):
        no_of_tokens=len(self.token)
        len_of_corpus=len(self.split_corpus)
        self.matrix = np.zeros((no_of_tokens, no_of_tokens), dtype=float)
        for i in tqdm(range(len_of_corpus-1),desc="Processing"):
            x=self.find_indice(self.split_corpus[i])
            x_plus_1=self.find_indice(self.split_corpus[i+1])
            self.matrix[x,x_plus_1]+=1
        
        row_sums = self.matrix.sum(axis=1)

        # Divide each element in a row by the sum of that row
        self.normalized_matrix = self.matrix / row_sums[:, np.newaxis]
        
        print("Matrix Calcualted")
