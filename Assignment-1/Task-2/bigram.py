import re
from tqdm import tqdm
import numpy as np
import pickle


class BigramLM:
    def __init__(self, corpus):
        self.corpus = re.sub("\n", " ", corpus)
        self.split_corpus = self.corpus.split()
        self.token = set()
        self.token_to_index = {}
        self.normalized_matrix = None
        self.matrix = None
        self.laplace_matrix = None
        self.kn_matrix = None
        self.emotions = pickle.load(open("pickle_files/emotions.pkl", "rb"))
        self.emotion_labels = {
            "sadness": 0,
            "joy": 1,
            "love": 2,
            "anger": 3,
            "fear": 4,
            "surprise": 5,
        }
        print("Model initialized 🟢")

    def set_laplace_matrix(self):
        row_sums = self.matrix.sum(axis=1)
        self.laplace_matrix = (self.matrix + 1) / (row_sums[:, np.newaxis] + len(self.token))
        return

    # def set_kn_matrix(self, d=0.75):

    #     bigram_matrix = self.matrix
    #     unigram_count = bigram_matrix.sum(axis=1)

    #     unique_bigrams = np.where(bigram_matrix != 0, 1, 0).sum()

    #     alpha = []
    #     continuation = []
    #     for row in range(len(bigram_matrix)):
    #         alpha.append(d * np.where(bigram_matrix[row] != 0, 1, 0).sum() / unigram_count[row])

    #     context_count = []
    #     for col in range(len(bigram_matrix[0])):
    #         context_count.append(np.where(bigram_matrix[:, col] != 0, 1, 0).sum())
    #         continuation.append(context_count[col] / unique_bigrams)

    #     self.kn_matrix = np.zeros((len(self.token), len(self.token)), dtype=float)

    #     for row in tqdm(range(len(bigram_matrix)), desc="Calculating Kneser-Ney Matrix..."):
    #         for col in range(len(bigram_matrix[row])):
    #             self.kn_matrix[row][col] = (
    #                 max(bigram_matrix[row][col] - d, 0) / unigram_count[row]
    #             )
    #             self.kn_matrix[row][col] += alpha[row] * continuation[col]

    #     return 

    def set_kn_matrix(self, d=0.75):
        print("Calculating Kneser-Ney matrix...")

        unigram_counts = self.matrix.sum(axis=1)
        unique_bigrams = np.count_nonzero(self.matrix)

        alpha = d * np.count_nonzero(self.matrix, axis=1) / unigram_counts
        continuation = np.count_nonzero(self.matrix, axis=0) / unique_bigrams

        self.kn_matrix = np.maximum(self.matrix - d, 0) / unigram_counts[:, np.newaxis] + alpha[:, np.newaxis] * continuation

        print("Kneser-Ney matrix calculated 🟢")

        return

    def get_kn_matrix(self):
        return self.kn_matrix

    def get_normal_matrix(self):
        return self.normalized_matrix

    def get_count_matrix(self):
        return self.matrix

    def get_laplace_matrix(self):
        return self.laplace_matrix

    def get_tokens(self):
        return self.ordered_tokens

    def get_corpus(self):
        return self.split_corpus

    def get_sum(self):
        return self.matrix.sum(axis=1)

    # May need this if we want to work for a worse corpus
    def check_remove_punctuation(self):
        pass

    def set_tokens(self):
        for i in self.split_corpus:
            if i not in self.token:
                self.token.add(i)

        self.ordered_tokens = sorted(list(self.token))

        for i in range(len(self.ordered_tokens)):
            self.token_to_index[self.ordered_tokens[i]] = i
        print("Tokens set 🟢")
        return

    def find_index(self, token):
        return self.token_to_index[token]

    def calculate_bigrams(self):
        no_of_tokens = len(self.token)
        len_of_corpus = len(self.split_corpus)
        self.matrix = np.zeros((no_of_tokens, no_of_tokens), dtype=float)
        for i in tqdm(range(len_of_corpus - 1), desc="Populating bigram matrix..."):
            x = self.find_index(self.split_corpus[i])
            x_plus_1 = self.find_index(self.split_corpus[i + 1])
            self.matrix[x, x_plus_1] += 1

        row_sums = self.matrix.sum(axis=1)
        self.normalized_matrix = self.matrix / row_sums[:, np.newaxis]

        print("All matrices calculated 🟢")

    def get_emotion_matrix(self, matrix, emotion, alpha=1, beta=1):

        emotion_matrix = matrix.copy()
        tokens = self.get_tokens()
        index = self.emotion_labels[emotion]

        for i, token in tqdm(enumerate(tokens), desc="Generating " + emotion + " matrix"):
            for j, token2 in enumerate(tokens):
                if self.get_count_matrix()[i][j] > 0:
                    emotion_matrix[i][j] = (alpha * emotion_matrix[i][j]
                        + beta * np.around(self.emotions[(token, token2)][index]["score"]))

        with open(f"pickle_files/{emotion}.pkl", "wb") as f:
            pickle.dump(emotion_matrix, f)

        return emotion_matrix

    def generate_sentences(self, matrix, emotion, alpha, beta, word_limit=10, no_of_sentences=50):
        
        emotion_matrix = self.get_emotion_matrix(matrix, emotion, alpha, beta)
        normalized_emotion_matrix = emotion_matrix / emotion_matrix.sum(axis=1, keepdims=True)
        sentences = []
        for _ in tqdm(range(no_of_sentences), desc="Generating sentences"):
            start_token = "$"
            sentence = []
            index = self.find_index(start_token)
            end = False
            for _ in range(word_limit):
                sampled_indices = np.random.choice(
                    self.get_tokens(), size=1, p=normalized_emotion_matrix[index]
                )
                while(sampled_indices[0]=="$"):
                    ind = self.find_index(sampled_indices[0])
                    if normalized_emotion_matrix[index][ind] == 1:
                        end = True
                        break
                    sampled_indices = np.random.choice(self.get_tokens(), size=1, p=normalized_emotion_matrix[index])
                if end == True:
                    break
                sentence.append(sampled_indices[0])
                index = self.find_index(sampled_indices[0])    
            sentence = " ".join(sentence)
            sentences.append(sentence)

        with open(f"emotion_text/gen_{emotion}.txt", "w") as file:
            for sentence in sentences:
                file.write(sentence + "\n")

        print("Sentences generated and stored 🟢")
        return sentences
