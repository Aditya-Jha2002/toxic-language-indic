import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression
from src.utils import Utils
from src.build_features import BuildFeatures 
import argparse
import joblib
import json

class Trainer:
    """Trains a model on the given dataset"""
    def __init__(self, config_path):
        self.config_path = config_path

        config = Utils().read_params(config_path)

        self.clean_data_path = config["clean_dataset"]["clean_folds_path"]
        self.model_dir = config["model_dir"]
        self.tfv_artifact_path = config["build_features"]["tfv_artifact_path"]

        self.C = config["estimators"]["LogisticRegression"]["params"]["C"]
        self.l1_ratio = config["estimators"]["LogisticRegression"]["params"]["l1_ratio"]
        self.random_state = config["base"]["random_state"]

        self.scores_file = config["reports"]["scores_cv"]
        self.params_file = config["reports"]["params_cv"]

    def train_and_evaluate(self):
        """Train the model and evaluate the model performance"""
        running_accuracy, running_f1, running_roc_auc, running_log_loss = [], [], [], []

        for i in range(1, 6):
            (accuracy, f1, roc_auc, log_loss_score) = self._train_one_fold(i - 1)

            running_accuracy.append(float(accuracy))
            running_f1.append(float(f1))
            running_roc_auc.append(float(roc_auc))
            running_log_loss.append(float(log_loss_score))

        running_accuracy = sum(running_accuracy)/5
        running_f1 = sum(running_f1)/5
        running_roc_auc = sum(running_roc_auc)/5
        running_log_loss = sum(running_log_loss)/5

        print("-" * 50)
        print("Logistic Regression Model (C=%f, l1_ratio=%f):" % (self.C, self.l1_ratio))
        print(f"  ACCURACY: {running_accuracy}")
        print(f"  F1: {running_f1}")  
        print(f"  ROC AUC: {running_roc_auc}")
        print(f"  LOG LOSS: {running_log_loss}")
        print("-" * 50)
        print("-" * 50)


    #####################################################
    # Log Parameters and Scores for the deployed modle

        with open(self.scores_file, "w") as f:
            scores = {
                "accuracy_score": running_accuracy,
                "f1_score": running_f1,
                "roc_auc_score": running_roc_auc,
                "log_loss": running_log_loss
            }
            json.dump(scores, f, indent=4)

        with open(self.params_file, "w") as f:
            params = {
                "C": self.C,
                "l1_ratio": self.l1_ratio,
            }
            json.dump(params, f, indent=4)
    #####################################################

    def _train_one_fold(self, fold_num):
        print(f"Training fold {fold_num} ...")

        store_tfv_artifact = False
        if fold_num == 0:
            store_tfv_artifact = True
        xtrain_tfv, ytrain, xvalid_tfv, yvalid = BuildFeatures(self.config_path).build_features_train(fold_num, store_tfv_artifact)
        
        clf = LogisticRegression(
            C=self.C,
            l1_ratio=self.l1_ratio,
            solver="liblinear",
            random_state=self.random_state)

        clf.fit(xtrain_tfv, ytrain)
        preds = clf.predict(xvalid_tfv)
        pred_proba = clf.predict_proba(xvalid_tfv)

        (accuracy, f1, roc_auc, log_loss_score) = self._eval_metrics(yvalid, preds, pred_proba)
        
        print("-" * 50)
        print(f"  Fold {fold_num} score:")
        print(f"  ACCURACY: {accuracy}")
        print(f"  F1: {f1}")
        print(f"  ROC AUC: {roc_auc}")
        print(f"  LOG LOSS: {log_loss_score}")
        print("-" * 50)

        return accuracy, f1, roc_auc, log_loss_score

    def _eval_metrics(self, actual, pred, pred_proba):
        """ Takes in the ground truth labels, predictions labels, and prediction probabilities.
            Returns the accuracy, f1, auc_roc, log_loss scores.
        """
        accuracy = accuracy_score(actual, pred)
        f1 = f1_score(actual, pred)
        roc_auc = roc_auc_score(actual, pred_proba[:, 1])
        log_loss_score = log_loss(actual, pred_proba)
        return accuracy, f1, roc_auc, log_loss_score

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    Trainer(config_path=parsed_args.config).train_and_evaluate()