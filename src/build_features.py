# -*- coding: utf-8 -*-
import json
import os
import nltk
import numpy as np
import pandas as pd
from src.utils import Utils
from inltk.inltk import tokenize
from nltk.tokenize import word_tokenize

class BuildFeatures:
    """BuildFeatures class to take in train and validation features and perform feature engineering"""

    def __init__(self, config_path):
        config = Utils().read_params(config_path)
        self.langs = config["base"]["langs"]
        self.clean_dir = config["clean_dataset"]["clean_dir"]
        self.en_fasttext_path = config["build_features"]["en_fasttext_path"]
        self.hi_fasttext_path = config["build_features"]["hi_fasttext_path"]
        self.ta_fasttext_path = config["build_features"]["ta_fasttext_path"]

    def build_features_train(self, fold_num, lang):
        """Performs feature engineering to the folds data from (../processed) into
        features ready to be trained by a model (returned in the function).
        """
        # Load the clean data
        df = pd.read_csv(os.path.join(self.clean_dir, f"{lang}_folds.csv"))

        xtrain = df[df["kfold"] != fold_num]["comment_text"]
        ytrain = df[df["kfold"] != fold_num]["toxic"]

        xvalid = df[df["kfold"] == fold_num]["comment_text"]
        yvalid = df[df["kfold"] == fold_num]["toxic"]

        xtrain_ft = self._build_features_df(xtrain, lang)
        xvalid_ft = self._build_features_df(xvalid, lang)

        return xtrain_ft, ytrain, xvalid_ft, yvalid

    def build_features_test(self, lang):
        """Performs feature engineering to the test data from (../processed) into
        features ready to be trained by a model (returned in the function).
        """
        # Load the clean data
        df = pd.read_csv(os.path.join(self.clean_dir, f"{lang}_test.csv"))

        xtest = df["comment_text"]
        ytest = df["toxic"]

        # Fitting TF-IDF to both training and test sets (semi-supervised learning)
        xtest_tfv = self.build_features_df(xtest, lang)

        return xtest_tfv, ytest

    def _build_features_df(self, df, lang: str):
        """Performs feature engineering to the dataframe given in the arguments into
        features ready to be trained by a model (returned in the function).
        """
        if lang =="en":
            fasttext_path = self.en_fasttext_path
            tokenizer = word_tokenize
        if lang =="hi":
            fasttext_path = self.hi_fasttext_path
            tokenizer = tokenize
        if lang =="ta":
            fasttext_path = self.ta_fasttext_path
            tokenizer = tokenize

        # Load the fasttext embedding
        print("Loading embeddings")
        embeddings = self._load_vectors(fasttext_path)
        print("Done loading embeddings")

        # create sentence embeddings
        print("Creating sentence vectors") 
        vectors = []
        for text in df.comment_text.values:
            vectors.append( 
                self._sentence_to_vec(
                s = text,
                embedding_dict = embeddings, stop_words = [],
                tokenizer = tokenizer, lang = lang
            ) )

        return vectors 

    def _build_features_text(self, text: str, lang: str):
        """Runs feature engineering scripts to turn the text given as input,
        into features ready to be trained by a model (returned in the function).
        """
        if lang =="en":
            fasttext_path = self.en_fasttext_path
            tokenizer = word_tokenize
        if lang =="hi":
            fasttext_path = self.hi_fasttext_path
            tokenizer = self._tokenize
        if lang =="ta":
            fasttext_path = self.ta_fasttext_path
            tokenizer = self._tokenize

        # Load the fasttext embedding
        print("Loading embeddings")
        embeddings = self._load_vectors(fasttext_path)
        print("Done loading embeddings")

        # create sentence embeddings
        print("Creating sentence vectors") 
        vectors = []
        vectors.append( 
                self._sentence_to_vec(
                s = text,
                embedding_dict = embeddings, stop_words = [],
                tokenizer = tokenizer, lang = lang
            ) )

        return vectors

    def _load_vectors(self, fname): 
        embeddings = json.load(open(fname))
        return embeddings
    
    def _sentence_to_vec(s, embedding_dict, stop_words, tokenizer, lang): 
        """
        Given a sentence and other information,
        this function returns embedding for the whole sentence :param s: sentence, string
        :param embedding_dict: dictionary word:vector
        :param stop_words: list of stop words, if any
        :param tokenizer: a tokenization function
        """
        # convert sentence to string and lowercase it
        words = str(s).lower()
        # tokenize the sentence
        words = tokenizer(words)
        # remove stop word tokens
        words = [w for w in words if not w in stop_words] 
        # keep only alpha-numeric tokens
        words = [w for w in words if w.isalpha()]
        # initialize empty list to store embeddings
        M = []
        for w in words:
        # for every word, fetch the embedding from # the dictionary and append to list of
        # embeddings
            if w in embedding_dict:
                M.append(embedding_dict[w])

        # if we dont have any vectors, return zeros
        if len(M) == 0:
            return np.zeros(300)
        # convert list of embeddings to array
        M = np.array(M)
        # calculate sum over axis=0
        v = M.sum(axis=0)
        # return normalized vector
        return v / np.sqrt((v ** 2).sum())

    def _tokenize(self, text):
        """
        Tokenizes a text into words.
        """
        return text.split()