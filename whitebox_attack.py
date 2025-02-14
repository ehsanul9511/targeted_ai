import torch as ch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt

def neuron_output(data, MLP):
    """
        For a given model and some data, get output for each layer's activations < layer.
        If get_all is True, return all activations unconditionally.
    """
    X = ch.tensor(data, dtype=ch.float64)
    L = X
    all = []
    # calculate the depth of the network
    layer = len(MLP.coefs_)
    # print(layer)
    for i in range(layer):
        L = ch.relu(ch.matmul(L, ch.tensor(MLP.coefs_[i])) + ch.tensor(MLP.intercepts_[i]))
        # all.append(L)
        all = all + [L]

    all = ch.cat(all, dim=1)
    return all.detach().numpy()


def make_neuron_output_data(ds, df, MLP, y_columns=None):
    df = df.drop(y_columns, axis=1)
    X_with_sensitive_positive = df.copy()
    positive_value = ds.ds.meta['sensitive_positive']
    negative_value = [val for val in ds.ds.meta['sensitive_values'] if val != positive_value][0]
    X_with_sensitive_positive[f'{ds.ds.meta["sensitive_column"]}_{positive_value}'] = 1
    X_with_sensitive_positive[f'{ds.ds.meta["sensitive_column"]}_{negative_value}'] = 0
    X_with_sensitive_negative = df.copy()
    X_with_sensitive_negative[f'{ds.ds.meta["sensitive_column"]}_{positive_value}'] = 0
    X_with_sensitive_negative[f'{ds.ds.meta["sensitive_column"]}_{negative_value}'] = 1

    X_neuron_output_positive = neuron_output(X_with_sensitive_positive.to_numpy(), MLP)
    X_neuron_output_negative = neuron_output(X_with_sensitive_negative.to_numpy(), MLP)

    X_neuron_output = np.concatenate((X_neuron_output_positive, X_neuron_output_negative), axis=1)

    X_neuron_output = pd.DataFrame(X_neuron_output, index=df.index)

    return X_neuron_output




def roc_curve_plot(fpr, tpr):
    # Calculate the area under the ROC curve (AUC)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

def get_LOMIA_case_1_correct_examples(ds, X_train):
    attack_df = pd.read_csv(f'<PATH_TO_DATASET>/{ds.ds.name}_attack.csv')
    attack_df.index = attack_df['Unnamed: 0']
    attack_df.index.name = None
    attack_df = attack_df.drop('Unnamed: 0', axis=1)
    return (X_train.loc[attack_df.index][ds.ds.meta['sensitive_column'] + "_" + ds.ds.meta['sensitive_positive']] == attack_df[ds.ds.meta['sensitive_column'] + "_" + ds.ds.meta['sensitive_positive']]).to_numpy().nonzero()[0]


class Top10CorrNeurons(nn.Module):
    def __init__(self, top_10_neuron_mean, top_10_neuron_std, corrs_top_10, corrs_top_10_vals):
        super(Top10CorrNeurons, self).__init__()
        self.top_10_neuron_mean = top_10_neuron_mean
        self.top_10_neuron_std = top_10_neuron_std
        self.corrs_top_10 = corrs_top_10
        self.corrs_top_10_vals = corrs_top_10_vals/np.sum(corrs_top_10_vals)
        self.threshold = 0.5

    def forward(self, X_neuron):
        # normalize the top 10 neurons
        X_neuron_norm = (X_neuron[:, self.corrs_top_10] - self.top_10_neuron_mean) / self.top_10_neuron_std
        # multiply by the correlation values
        X_neuron_norm_corr = X_neuron_norm * self.corrs_top_10_vals
        # sum the neurons
        X_neuron_norm_corr_sum = ch.sum(X_neuron_norm_corr, dim=1)
        return X_neuron_norm_corr_sum
    
def wb_corr_attacks(X_neuron, y):
    X_neuron = X_neuron.to_numpy()
    X_neuron_pos_val = X_neuron[:, :X_neuron.shape[1]//2]
    y_wb_att = y.to_numpy().ravel().astype(np.float32)

    corrs = np.corrcoef(X_neuron_pos_val[:, :], y_wb_att, rowvar=False)[-1,:-1]
    # replace nan values with -inf
    corrs = np.nan_to_num(corrs, nan=-np.inf)
    # find the top 10 correlated neurons
    corrs_top_10 = np.argsort(corrs)[-10:]
    corrs_top_10_vals = corrs[corrs_top_10]

    # we need to scale X_neuron by the mean and std of the top 10 correlated neurons
    top_10_neuron_mean = np.mean(X_neuron_pos_val[:, corrs_top_10], axis=0)
    top_10_neuron_std = np.std(X_neuron_pos_val[:, corrs_top_10], axis=0)

    # create the model
    top_10_corr_neurons_model = Top10CorrNeurons(top_10_neuron_mean, top_10_neuron_std, corrs_top_10, corrs_top_10_vals)

    y_pred = top_10_corr_neurons_model(ch.from_numpy(X_neuron).float()).detach().numpy()

    # get the best threshold
    fpr, tpr, thresholds = roc_curve(y_wb_att, y_pred)
    best_threshold = thresholds[np.argmax(tpr - fpr)]
    top_10_corr_neurons_model.threshold = best_threshold

    return top_10_corr_neurons_model