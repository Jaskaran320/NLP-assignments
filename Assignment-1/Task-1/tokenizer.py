import re
from collections import defaultdict


class Tokenizer:
    def __init__(self):
        self.orig_vocab = {}
        self.vocab = {}
        self.left_vocab = {}
        self.merge_rules = []
        for i in range(97, 123):
            self.left_vocab[chr(i)] = 1

    def learn_vocabulary(self, corpus, num_merges):
        self.orig_vocab = self.get_unigrams(corpus)

        for _ in range(num_merges):
            pairs = self.get_stats(self.orig_vocab)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            self.merge_rules.append(best_pair)
            self.orig_vocab = self.merge_vocab(best_pair, self.orig_vocab)

        print(f"Vocabulary learned successfully with {len(self.merge_rules)} merge rules")
        self.orig_vocab.update(self.left_vocab)

        for key in self.orig_vocab.keys():
            if " " in key:
                for char in key.split(" "):
                    if char not in self.vocab:
                        self.vocab[char] = 1
            else:
                self.vocab[key] = 1

        self.vocab = dict(sorted(self.vocab.items(), key=lambda item: (len(item[0]), item[0])))

    def tokenize(self, sample):
        sample = sample.replace(" ", "$")
        sample = sample + "$"
        tokens = []
        for word in sample.split():
            tokens.extend(self.split_word(word))

        return tokens

    def get_unigrams(self, corpus):
        unigrams = defaultdict(int)
        for word in corpus.split():
            word = word + "$"
            token = " ".join(word)
            unigrams[token] += 1

        return unigrams

    def get_stats(self, vocab):
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq

        return pairs

    def merge_vocab(self, pair, v_in):
        v_out = {}
        bigram = re.escape(" ".join(pair))
        p = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
        for word in v_in:
            w_out = p.sub("".join(pair), word)
            v_out[w_out] = v_in[word]
            self.left_vocab[w_out] = v_in[word]

        return v_out

    def split_word(self, word):
        if word in self.vocab:
            return [word]
        subwords = []
        start = 0
        while start < len(word):
            end = start + 1
            max_subword = None
            while end <= len(word):
                subword = word[start:end]
                if subword in self.vocab:
                    max_subword = subword
                end += 1
            if max_subword is not None:
                subwords.append(max_subword)
                start += len(max_subword)
            else:
                subwords.append(word[start])
                start += 1

        return subwords

    # alternate split_word, can be used if the corpus has all 26 letters

    # def split_word(self, word):
    #     if word in self.vocab:
    #         return [word]
    #     subwords = []
    #     start = 0
    #     while start < len(word):
    #         end = len(word)
    #         while end > start:
    #             subword = word[start:end]
    #             if subword in self.vocab:
    #                 subwords.append(subword)
    #                 start = end
    #                 break
    #             end -= 1
    #         if start != end:
    #             subwords.append(word[start])
    #             start += 1

    #     return subwords

    def write_all_tokens(self, tokens_path):
        with open(tokens_path, "w") as file:
            for word in self.vocab.keys():
                file.write(word + "\n")

    def write_merge_rules(self, merge_rules_path):
        with open(merge_rules_path, "w") as file:
            for rule in self.merge_rules:
                file.write(",".join(rule) + "\n")

    def write_samples(self, samples_path, samples):
        with open(samples_path, "w") as file:
            for sample in samples:
                tokens = self.tokenize(sample)
                file.write(",".join(tokens) + "\n")
