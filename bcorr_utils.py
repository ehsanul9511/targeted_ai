from tqdm import tqdm
import os
import data_utils
import model_utils
from attack_utils import get_CSMIA_case_by_case_results, CSMIA_attack, LOMIA_attack
from data_utils import oneHotCatVars, filter_random_data_by_conf_score
from experiment_utils import MIAExperiment
from disparity_inference_utils import get_confidence_array, draw_confidence_array_scatter, get_indices_by_group_condition, get_corr_btn_sens_and_out_per_subgroup, get_slopes, get_angular_difference, calculate_stds, get_mutual_info_btn_sens_and_out_per_subgroup
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network._base import ACTIVATIONS
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from fairlearn.metrics import equalized_odds_difference, demographic_parity_difference
import matplotlib.pyplot as plt
import seaborn as sns
import tabulate
import pickle
# import utils
import copy

from sklearn.neural_network import MLPClassifier

class MLPClassifierFC(MLPClassifier):
    def fit(self, X, y, sample_weight=None):
        """
            Fit the model to the given data.
        """
        if sample_weight is not None:
            # resample data according to sample weights
            n_samples = X.shape[0]
            sample_weight = np.asarray(sample_weight)/np.sum(sample_weight)
            sample_idxs = np.random.choice(n_samples, n_samples, p=sample_weight)
            X = X.iloc[sample_idxs]
            y = y[sample_idxs]
            
        return super().fit(X, y)


def bcorr_sampling(experiment, X_train, y_tr, y_tr_onehot, subgroup_col_name, p=-0.1):
    sensitive_col_name, sensitive_positive, y_column = experiment.ds.ds.meta['sensitive_column'], experiment.ds.ds.meta['sensitive_positive'], experiment.ds.ds.meta['y_column']

    df = X_train.copy()
    df[y_column] = y_tr.ravel()
    subgroup_values = [col.split('_')[-1] for col in df.columns if col.startswith(subgroup_col_name)]
    n = [df[df[f"{subgroup_col_name}_{val}"]==1][[f'{sensitive_col_name}_{sensitive_positive}', y_column]].value_counts().to_numpy().min() * 4 for val in subgroup_values]
    p = [p] * len(subgroup_values)

    print(n)

    sample_indices = experiment.ds.ds.sample_indices_matching_correlation(X_train, y_tr, p=p, n=n, subgroup_col_name=subgroup_col_name, random_state=experiment.random_state)

    X_train_balanced_corr = X_train.loc[sample_indices].reset_index(drop=True)
    y_tr_balanced_corr = y_tr[sample_indices]
    y_tr_onehot_balanced_corr = y_tr_onehot[sample_indices]

    return X_train_balanced_corr, y_tr_balanced_corr, y_tr_onehot_balanced_corr


def evaluate(experiment, clf, X_train, y_tr, X_test, y_te, subgroup_col_name):
    # y_te_pred = np.argmax(clf.predict_proba(X_test), axis=1)
    if isinstance(clf, MLPClassifier):
        y_te_pred = np.argmax(clf.predict_proba(X_test), axis=1)
    else:
        y_te_pred = clf.predict(X_test)

    subgroup_oh_cols = [col for col in X_train.columns if subgroup_col_name in col]
    subgroup_vals_tr = X_train[subgroup_oh_cols].to_numpy().argmax(axis=1)
    subgroup_vals_te = X_test[subgroup_oh_cols].to_numpy().argmax(axis=1)

    sensitive_columns = [f'{experiment.ds.ds.meta["sensitive_column"]}_{i}' for i in experiment.ds.ds.meta["sensitive_values"]]
    sens_val_ground_truth = X_train[sensitive_columns].idxmax(axis=1).str.replace(f'{experiment.ds.ds.meta["sensitive_column"]}_', '')
    sens_val_ground_truth = sens_val_ground_truth.astype(experiment.ds.ds.original_df[experiment.ds.ds.meta["sensitive_column"]].dtype)
    sens_val_ground_truth = np.array([{x: i for i, x in enumerate(experiment.ds.ds.meta["sensitive_values"])}[val] for val in sens_val_ground_truth])

    sens_pred, case_indices = CSMIA_attack(clf, X_train, y_tr, experiment.ds.ds.meta)
    sens_pred_LOMIA = LOMIA_attack(experiment, clf, X_train, y_tr, experiment.ds.ds.meta)


    correct_indices = (sens_pred == sens_val_ground_truth)
    correct_indices_LOMIA = (sens_pred_LOMIA == sens_val_ground_truth)
    
    num_of_subgroups = len(subgroup_oh_cols)
    perf_dict = {
        'ASRD_CSMIA': round(100 * np.ptp([correct_indices[subgroup_vals_tr==i].mean() for i in range(num_of_subgroups)]), 2),
        'ASRD_LOMIA': round(100 * np.ptp([correct_indices_LOMIA[subgroup_vals_tr==i].mean() for i in range(num_of_subgroups)]), 2),
        'EOD': round(equalized_odds_difference(y_te.ravel(), y_te_pred, sensitive_features=subgroup_vals_te), 4),
        'DPD': round(demographic_parity_difference(y_te.ravel(), y_te_pred, sensitive_features=subgroup_vals_te), 4),
        'MA': 100 * accuracy_score(y_te.ravel()[:], y_te_pred[:])
    }

    return perf_dict



