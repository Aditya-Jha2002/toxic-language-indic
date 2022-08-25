# -*- coding: utf-8 -*-
import os
import argparse
import pandas as pd
from src.utils import Utils
from sklearn.model_selection import train_test_split, StratifiedKFold

class CVFoldsDataset:
    """Split the dataset into kfolds dataset and test dataset"""

    def __init__(self, config_path):
        config = Utils().read_params(config_path)
        self.merge_dir = config["load_dataset"]["merge_dir"]
        self.fold_num = config["split_dataset"]["fold_num"]
        self.test_size = config["split_dataset"]["test_size"]
        self.split_dir = config["split_dataset"]["split_dir"]
        self.random_state = config["base"]["random_state"]

    def cv_folds_dataset(self):
        """Runs scripts to load the raw data from (../raw) into
        fold datasets ready to be further cleaned on (saved in ../interim),
        and test dataset ready to be further cleaned on (saved in ../interim)
        """
        for lang in ["english", "tamil", "hindi"]:
            self._folds_and_test_dataset(lang)

    def _folds_and_test_dataset(self, lang):
        df = pd.read_csv(os.path.join(self.merge_dir, f"{lang}_merge.csv"))

        # Create test set
        df, test = train_test_split(
            df, test_size=self.test_size, random_state=self.random_state
        )

        # Create folds
        df_folds = self._create_folds(df)

        # Save the folds data
        df_folds.to_csv(os.path.join(self.split_dir, f"{lang[:2]}_folds.csv"), index=False)
        # Save the test data
        test.to_csv(os.path.join(self.split_dir, f"{lang[:2]}_test.csv"), index=False)

    def _create_folds(self, data):
        """Create folds for cross-validation"""

        # create the new kfold column
        data["kfold"] = -1

        # randomize the rows of the data
        data = data.sample(frac=1).reset_index(drop=True)

        # initiate the kfold class from model_selection module
        kf = StratifiedKFold(n_splits=self.fold_num)

        # fill the new kfold column
        for f, (t_, v_) in enumerate(kf.split(X=data, y=data.toxic.values)):
            data.loc[v_, "kfold"] = f

        return data


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()

    CVFoldsDataset(config_path=parsed_args.config).cv_folds_dataset()
