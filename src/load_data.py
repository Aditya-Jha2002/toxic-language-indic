# -*- coding: utf-8 -*-
import os
import re
import argparse
from src import utils
import string
import nltk
import emoji
import pandas as pd
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords

nltk.download("stopwords")


class DataLoader:
    """DataLoader class to load the data, and get it to the proper format"""

    def __init__(self, config_path):
        config = utils.Utils().read_params(config_path)
        self.base_dir = config["load_dataset"]["base_dir"]
        self.en_train_path = config["load_dataset"]["en_train_path"]
        self.hi_train_path = config["load_dataset"]["hi_train_path"]
        self.hi_val_path = config["load_dataset"]["hi_val_path"]
        self.hi_test_path = config["load_dataset"]["hi_test_path"]
        self.ta_train_path = config["load_dataset"]["ta_train_path"]
        self.ta_val_path = config["load_dataset"]["ta_val_path"]
        self.ta_test_path = config["load_dataset"]["ta_test_path"]
        self.merge_dir = config["load_dataset"]["merge_dir"]
        self.en_merge_path = config["load_dataset"]["en_merge_path"]
        self.hi_merge_path = config["load_dataset"]["hi_merge_path"]
        self.ta_merge_path = config["load_dataset"]["ta_merge_path"]

    def load_dataset(self):
        """Runs preprocessing scripts to turn data given from (../interim) into
        cleaned and pre-processed data ready to be feature engineered on (saved in ../processed).
        """
        # Preprocess the text
        df_tamil = self._merge_datasets([self.ta_train_path, self.ta_val_path, self.ta_test_path])
        df_hindi = self._merge_datasets([self.hi_train_path, self.hi_val_path, self.hi_test_path])
        df_english = pd.read_csv(os.path.join(self.base_dir, self.en_train_path))

        df_english.drop(["id", "severe_toxic", "obscene", "threat", "insult", "identity_hate"], axis=1, inplace=True)

        df_tamil = self._format_dataset(df_tamil)
        df_hindi = self._format_dataset(df_hindi)
        
        # Save the clean data
        df_english.to_csv(os.path.join(self.merge_dir, self.en_merge_path), sep=",", index=False)
        df_hindi.to_csv(os.path.join(self.merge_dir, self.hi_merge_path), sep=",", index=False)
        df_tamil.to_csv(os.path.join(self.merge_dir, self.ta_merge_path), sep=",", index=False)

    def _merge_datasets(self, df_path_list):
        """Merges the dataframes in the list"""
        df = pd.DataFrame()
        for df_path in df_path_list:
            df = df.append(pd.read_csv(os.path.join(self.base_dir,df_path)))
        return df

    def _format_dataset(self, df):
        """Rename the columns etc."""
        df = df.rename(columns={"text": "comment_text", "label": "toxic"})
        df["toxic"] = df["toxic"].map({0: 1, 1: 0})
        return df

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()

    DataLoader(config_path=parsed_args.config).load_dataset()