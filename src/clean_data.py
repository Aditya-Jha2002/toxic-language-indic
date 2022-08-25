# -*- coding: utf-8 -*-
import re
import os
import pandas as pd
import argparse
from src import utils
import string

import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords

nltk.download("stopwords")


class DataCleaner:
    """DataCleaner class to load the data, and preprocess it"""

    def __init__(self, config_path):
        config = utils.Utils().read_params(config_path)
        self.split_dir = config["split_dataset"]["split_dir"]
        self.clean_dir = config["clean_dataset"]["clean_dir"]

    def clean_dataset(self, df_type: str):
        """Runs preprocessing scripts to turn data given from (../interim) into
        cleaned and pre-processed data ready to be feature engineered on (saved in ../processed).
        """
        for lang in ["en", "hi", "ta"]:
            for df_type in ["folds", "test"]:
                df = pd.read_csv(os.path.join(self.split_dir,f"{lang}_{df_type}.csv"))
                df["comment_text"] = df["comment_text"].apply(self._preprocess_text)
                df.to_csv(os.path.join(self.clean_dir,f"{lang}_{df_type}.csv"), index=False)

    def _preprocess_text(self, text: str) -> str:
        """Preprocess and clean text given"""
        # Remove weird spaces
        text = self._remove_space(text)

        # Remove emojis
        text = self._remove_emoji(text)  

        # Remove urls
        text = self._remove_urls(text)

        # Remove punctuations
        text = self._remove_punctuation(text)

        return text

    def _remove_space(self, text: str) -> str:
        """To remove weird spaces from text"""
        text = text.strip()
        text = text.split()
        return " ".join(text)

    def _remove_contractions(self, text: str) -> str:
        """To remove the contractions like shan't and convert them to shall not"""
        for key in utils.contractions.keys():
            text = text.replace(key, utils.contractions[key])
        return text

    def _remove_mentions(self, text: str) -> str:
        """To remove the mentions from text"""
        text = re.sub(r"@[^ ]+", "", text)
        return text

    def _remove_emoji(self, text: str) -> str:
        emoji_pattern = re.compile("["
                                u"\U0001F600-\U0001F64F"  # emoticons
                                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                u"\U00002500-\U00002BEF"  # chinese char
                                u"\U00002702-\U000027B0"
                                u"\U00002702-\U000027B0"
                                u"\U000024C2-\U0001F251"
                                u"\U0001f926-\U0001f937"
                                u"\U00010000-\U0010ffff"
                                u"\u2640-\u2642"
                                u"\u2600-\u2B55"
                                u"\u200d"
                                u"\u23cf"
                                u"\u23e9"
                                u"\u231a"
                                u"\ufe0f"  # dingbats
                                u"\u3030"
                                "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)

    def _remove_urls(self, text: str) -> str:
        """To remove the urls from text"""
        text = re.sub(r"http\S+", "", text)
        return text

    def _remove_stopwords(self, text: str) -> str:
        """To remove the stopwords"""
        STOPWORDS = set(stopwords.words("english"))
        text = [word for word in str(text).split() if word not in STOPWORDS]
        return " ".join(text)

    def _remove_punctuation(self, text: str) -> str:
        """To remove the punctuations like !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
        text = text.translate(str.maketrans("", "", string.punctuation))
        return text

    def _stem_words(self, text: str) -> str:
        """To stem the words"""
        lemmatizer = WordNetLemmatizer()
        text = [lemmatizer.lemmatize(word) for word in text.split()]
        return " ".join(text)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()

    DataCleaner(config_path=parsed_args.config).clean_dataset("train")
    DataCleaner(config_path=parsed_args.config).clean_dataset("test")
