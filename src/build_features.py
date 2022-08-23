# -*- coding: utf-8 -*-
from src.utils import Utils
from nltk.tokenize import word_tokenize
import nltk
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download("punkt")


class BuildFeatures:
    """BuildFeatures class to take in train and validation features and perform feature engineering"""

    def __init__(self, config_path):
        config = Utils().read_params(config_path)
        self.clean_folds_path = config["clean_dataset"]["clean_folds_path"]
        self.clean_test_path = config["clean_dataset"]["clean_test_path"]
        self.tfv_artifact_path = config["build_features"]["tfv_artifact_path"]

    def build_features_train(self, fold_num, store_tfv):
        """Performs feature engineering to the folds data from (../processed) into
        features ready to be trained by a model (returned in the function).
        """
        # Load the clean data
        df = Utils().get_data(self.clean_folds_path)

        df.fillna(" ", inplace=True)

        xtrain = df[df["kfold"] != fold_num]["comment_text"]
        ytrain = df[df["kfold"] != fold_num]["toxic"]

        xvalid = df[df["kfold"] == fold_num]["comment_text"]
        yvalid = df[df["kfold"] == fold_num]["toxic"]

        # Create a tfidf vectorizer
        tfv = TfidfVectorizer(
            min_df=3,
            max_features=None,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"\w{1,}",
            ngram_range=(1, 3),
            use_idf=1,
            smooth_idf=1,
            sublinear_tf=1,
            stop_words="english",
        )

        # Fitting TF-IDF to both training and test sets (semi-supervised learning)
        tfv.fit(list(xtrain) + list(xvalid))

        if store_tfv:
            pickle.dump(tfv, open(self.tfv_artifact_path, "wb"))

        xtrain_tfv = tfv.transform(xtrain)
        xvalid_tfv = tfv.transform(xvalid)

        return xtrain_tfv, ytrain, xvalid_tfv, yvalid

    def build_features_test(self):
        """Performs feature engineering to the test data from (../processed) into
        features ready to be trained by a model (returned in the function).
        """
        # Load the clean data
        df = Utils().get_data(self.clean_test_path)

        df.fillna(" ", inplace=True)

        xtest = df["comment_text"]
        ytest = df["toxic"]

        # Load TF-IDF
        tfv = pickle.load(open(self.tfv_artifact_path, "rb"))

        # Fitting TF-IDF to both training and test sets (semi-supervised learning)
        xtest_tfv = tfv.transform(list(xtest))

        return xtest_tfv, ytest

    def _build_features_text(self, text: str):
        """Runs feature engineering scripts to turn the text given as input,
        into features ready to be trained by a model (returned in the function).
        """
        tfv = pickle.load(open(self.tfv_artifact_path, "rb"))

        # Fitting TF-IDF to both training and test sets (semi-supervised learning)
        text_tfv = tfv.transform([text])
        return text_tfv
