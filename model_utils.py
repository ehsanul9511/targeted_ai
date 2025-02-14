import numpy as np
import pandas as pd
from numpy.random import RandomState
from tqdm import tqdm
from typing import List, Literal
import torch as ch
import torch.nn as nn
import os
# from utils import check_if_inside_cluster, make_affinity_features
from joblib import load, dump
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network._base import ACTIVATIONS
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, matthews_corrcoef, confusion_matrix
from imblearn.metrics import geometric_mean_score
import platform
from fairlearn.reductions import ExponentiatedGradient
from fairlearn.reductions._moments import ClassificationMoment

device = ch.device("cuda" if ch.cuda.is_available() else "mps" if platform.machine() == "arm64" else "cpu")


BASE_MODELS_DIR = "<PATH_TO_MODELS>"
ACTIVATION_DIMS = [32, 16, 8, 2]


class MLPClassifierFC(MLPClassifier):
    def fit(self, X, y, sample_weights=None):
        """
            Fit the model to the given data.
        """
        if sample_weights is not None:
            # resample data according to sample weights
            rng = RandomState(self.random_state)
            n_samples = X.shape[0]
            sample_idxs = rng.choice(n_samples, size=n_samples, replace=True, p=sample_weights)
            X = X[sample_idxs]
            y = y[sample_idxs]
            
        return super().fit(X, y)


class PortedMLPClassifier(nn.Module):
    def __init__(self, n_in_features=37, n_out_features=2):
        super(PortedMLPClassifier, self).__init__()
        layers = [
            nn.Linear(in_features=n_in_features, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=8),
            nn.ReLU(),
            nn.Linear(in_features=8, out_features=n_out_features),
            nn.Softmax(dim=1)
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x: ch.Tensor,
                latent: int = None,
                get_all: bool = False,
                detach_before_return: bool = True,
                on_cpu: bool = False):
        """
        Args:
            x: Input tensor of shape (batch_size, 42)
            latent: If not None, return only the latent representation. Else, get requested latent layer's output
            get_all: If True, return all activations
            detach_before_return: If True, detach the latent representation before returning it
            on_cpu: If True, return the latent representation on CPU
        """
        if latent is None and not get_all:
            return self.layers(x)

        if latent not in [0, 1, 2] and not get_all:
            raise ValueError("Invald interal layer requested")

        if latent is not None:
            # First three hidden layers correspond to outputs of
            # Model layers 1, 3, 5
            latent = (latent * 2) + 1
        valid_for_all = [1, 3, 5, 6]

        latents = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Append activations for all layers (post-activation only)
            if get_all and i in valid_for_all:
                if detach_before_return:
                    if on_cpu:
                        latents.append(x.detach().cpu())
                    else:
                        latents.append(x.detach())
                else:
                    if on_cpu:
                        latents.append(x.cpu())
                    else:
                        latents.append(x)
            if i == latent:
                if on_cpu:
                    return x.cpu()
                else:
                    return x

        return latents

    def predict_proba(self, x: ch.Tensor):
        return self.forward(x)

    
# def train_torch_model(model=None, X=None, y=None, epochs=100, lr=0.01):
#     """
#         Train PyTorch model on given data
#     """
#     if model is None:
#         model = PortedMLPClassifier(n_in_features=X.shape[1], n_out_features=y.shape[1])
#     model = model.to(device)
#     if X is None or y is None:
#         return model
#     loss_fn = nn.CrossEntropyLoss()
#     optimizer = ch.optim.Adam(model.parameters(), lr=lr)
#     X = ch.tensor(X, dtype=ch.float32).to(device)
#     y = ch.tensor(y, dtype=ch.long).to(device)
#     for epoch in range(epochs):
#         optimizer.zero_grad()
#         y_pred = model(X)
#         loss = loss_fn(y_pred, y)
#         loss.backward()
#         optimizer.step()
#     return model


def port_mlp_to_ch(clf):
    """
        Extract weights from MLPClassifier and port
        to PyTorch model.
    """
    nn_model = PortedMLPClassifier(n_in_features=clf.coefs_[0].shape[0],
                                   n_out_features=clf.coefs_[-1].shape[1])
    i = 0
    for (w, b) in zip(clf.coefs_, clf.intercepts_):
        w = ch.from_numpy(w.T).float()
        b = ch.from_numpy(b).float()
        nn_model.layers[i].weight = nn.Parameter(w)
        nn_model.layers[i].bias = nn.Parameter(b)
        i += 2  # Account for ReLU as well

    # nn_model = nn_model.cuda()
    nn_model = nn_model.to(device)
    return nn_model


def port_ch_to_mlp(nn_model, clf=None):
    """
        Extract weights from PyTorch model and port
        to MLPClassifier.
    """
    # if clf is None:
    #     clf = get_model()
    #     y_shape_1 = nn_model.layers[-1].weight.shape[0]
    #     dtype = nn_model.layers[-1].weight.dtype
    #     hidden_layer_sizes = [nn_model.layers[i].weight.shape[1] for i in range(0, len(nn_model.layers), 2)]
    #     layer_units = [nn_model.layers[0].weight.shape[0]] + hidden_layer_sizes + [y_shape_1]
    #     clf.set_params(hidden_layer_sizes=layer_units, activation='relu', solver='adam', alpha=0.0001)
    #     clf._initialize(np.zeros(1,y_shape_1), layer_units, dtype)

    for i, layer in enumerate(nn_model.layers):
        if i % 2 == 0:
            clf.coefs_[i // 2] = layer.weight.detach().cpu().numpy().T
            clf.intercepts_[i // 2] = layer.bias.detach().cpu().numpy()

    return clf

def proxy_train_mlp(X, y, epochs=100, lr=0.01, l1_reg=0.0):
    """
        Train PyTorch model on given data
    """
    nn_model = train_torch_model(model=None, X=X, y=y, epochs=epochs, lr=lr, l1_reg=l1_reg)
    clf = get_model(max_iter=1)
    clf.fit(X, y)
    clf = port_ch_to_mlp(nn_model, clf)
    return clf


def convert_to_torch(clfs):
    """
        Port given list of MLPClassifier models to
        PyTorch models
    """
    return np.array([port_mlp_to_ch(clf) for clf in clfs], dtype=object)


# def layer_output(data, MLP, layer=0, get_all=False):
#     """
#         For a given model and some data, get output for each layer's activations < layer.
#         If get_all is True, return all activations unconditionally.
#     """
#     L = data.copy()
#     all = []
#     for i in range(layer):
#         L = ACTIVATIONS['relu'](
#             np.matmul(L, MLP.coefs_[i]) + MLP.intercepts_[i])
#         if get_all:
#             all.append(L)
#     if get_all:
#         return all
#     return L


def layer_output(data, MLP, layer=0, get_all=False):
    """
        For a given model and some data, get output for each layer's activations < layer.
        If get_all is True, return all activations unconditionally.
    """
    X = ch.tensor(data, dtype=ch.float64)
    L = X
    all = []
    for i in range(layer):
        L = ch.relu(ch.matmul(L, ch.tensor(MLP.coefs_[i])) + ch.tensor(MLP.intercepts_[i]))
        if get_all:
            all.append(L)
    if get_all:
        return [L.detach().numpy() for L in all]
    return L.detach().numpy()


# Load models from directory, return feature representations
def get_model_representations(folder_path, label, first_n=np.inf,
                              n_models=1000, start_n=0,
                              fetch_models: bool = False,
                              shuffle: bool = True,
                              models_provided: bool = False):
    """
        If models_provided is True, folder_path will actually be a list of models.
    """
    if models_provided:
        models_in_folder = folder_path
    else:
        models_in_folder = os.listdir(folder_path)

    if shuffle:
        # Shuffle
        np.random.shuffle(models_in_folder)

    # Pick only N models
    models_in_folder = models_in_folder[:n_models]

    w, labels, clfs = [], [], []
    for path in tqdm(models_in_folder):
        if models_provided:
            clf = path
        else:
            clf = load_model(os.path.join(folder_path, path))
        if fetch_models:
            clfs.append(clf)

        # Extract model parameters
        weights = [ch.from_numpy(x) for x in clf.coefs_]
        dims = [w.shape[0] for w in weights]
        biases = [ch.from_numpy(x) for x in clf.intercepts_]
        processed = [ch.cat((w, ch.unsqueeze(b, 0)), 0).float().T
                     for (w, b) in zip(weights, biases)]

        # Use parameters only from first N layers
        # and starting from start_n
        if first_n != np.inf:
            processed = processed[start_n:first_n]
            dims = dims[start_n:first_n]

        w.append(processed)
        labels.append(label)

    labels = np.array(labels)

    w = np.array(w, dtype=object)
    labels = ch.from_numpy(labels)

    if fetch_models:
        return w, labels, dims, clfs
    return w, labels, dims


def get_model(max_iter=40,
              hidden_layer_sizes=(32, 16, 8),
              random_state=42,
              verbose=False,
              learning_rate='constant'):
    """
        Create new MLPClassifier model
    """
    clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                        max_iter=max_iter,
                        random_state=random_state,
                        verbose=verbose,
                        learning_rate=learning_rate)
    return clf


def get_models(folder_path, n_models=1000, shuffle=True):
    """
        Load models from given directory.
    """
    paths = os.listdir(folder_path)
    if shuffle:
        paths = np.random.permutation(paths)
    paths = paths[:n_models]

    models = []
    for mpath in tqdm(paths):
        model = load_model(os.path.join(folder_path, mpath))
        models.append(model)
    return models


def save_model(clf, path):
    dump(clf, path)


def load_model(path):
    return load(path)


def get_models_path(property, split, value=None):
    if value is None:
        return os.path.join(BASE_MODELS_DIR, property, split)
    return os.path.join(BASE_MODELS_DIR,  property, split, value)


def get_model_activation_representations(
        models: List[PortedMLPClassifier],
        data, label, detach: bool = True,
        verbose: bool = True):
    w = []
    iterator = models
    if verbose:
        iterator = tqdm(iterator)
    for model in iterator:
        activations = model(data, get_all=True,
                            detach_before_return=detach)
        # Skip last feature (logit)
        activations = activations[:-1]

        w.append([act.float() for act in activations])
    labels = np.array([label] * len(w))
    labels = ch.from_numpy(labels)

    # Make numpy object (to support sequence-based indexing)
    w = np.array(w, dtype=object)

    # Get dimensions of feature representations
    dims = [x.shape[1] for x in w[0]]

    return w, labels, dims


def make_activation_data(models_pos, models_neg, seed_data,
                         detach=True, verbose=True, use_logit=False):
    # Construct affinity graphs
    pos_model_scores = make_affinity_features(
        models_pos, seed_data,
        detach=detach, verbose=verbose,
        use_logit=use_logit)
    neg_model_scores = make_affinity_features(
        models_neg, seed_data,
        detach=detach, verbose=verbose,
        use_logit=use_logit)
    # Convert all this data to loaders
    X = ch.cat((pos_model_scores, neg_model_scores), 0)
    Y = ch.cat((ch.ones(len(pos_model_scores)),
                ch.zeros(len(neg_model_scores))))
    return X, Y


def make_affinity_feature(model, data, use_logit=False, detach=True, verbose=True):
    """
         Construct affinity matrix per layer based on affinity scores
         for a given model. Model them in a way that does not
         require graph-based models.
    """
    # Build affinity graph for given model and data
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    # Start with getting layer-wise model features
    model_features = model(data, get_all=True, detach_before_return=detach)
    layerwise_features = []
    for i, feature in enumerate(model_features):
        # Old (before 2/4)
        # Skip logits if asked not to use (default)
        # if not use_logit and i == (len(model_features) - 1):
            # break
        scores = []
        # Pair-wise iteration of all data
        for i in range(len(data)-1):
            others = feature[i+1:]
            scores += cos(ch.unsqueeze(feature[i], 0), others)
        layerwise_features.append(ch.stack(scores, 0))

    # New (2/4)
    # If asked to use logits, convert them to probability scores
    # And then consider them as-it-is (instead of pair-wise comparison)
    if use_logit:
        logits = model_features[-1]
        probs = ch.sigmoid(logits)
        layerwise_features.append(probs)

    concatenated_features = ch.stack(layerwise_features, 0)
    return concatenated_features


def make_affinity_features(models, data, use_logit=False, detach=True, verbose=True):
    all_features = []
    iterator = models
    if verbose:
        iterator = tqdm(iterator, desc="Building affinity matrix")
    for model in iterator:
        all_features.append(
            make_affinity_feature(
                model, data, use_logit=use_logit, detach=detach, verbose=verbose)
        )
    return ch.stack(all_features, 0)

def train_torch_model(model=None, X=None, y=None, epochs=100, lr=0.01, l1_reg=0.0):
    """
        Train PyTorch model on given data
    """
    if model is None:
        model = PortedMLPClassifier(n_in_features=X.shape[1], n_out_features=y.shape[1])
    model = model.to(device)
    if X is None or y is None:
        return model
    def l1_loss(model):
        loss = 0.0
        for param in model.parameters():
            loss += ch.sum(ch.abs(param))
        loss = ch.mean(loss)
        return loss
    loss_fn = nn.CrossEntropyLoss()
    optimizer = ch.optim.Adam(model.parameters(), lr=lr)
    X = ch.tensor(X, dtype=ch.float32).to(device)
    y = ch.tensor(np.argmax(y, axis=1), dtype=ch.long).to(device) # Convert multi-target tensor to class labels
    # create dataset and dataloader
    dataset = ch.utils.data.TensorDataset(X, y)
    dataloader = ch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    for epoch in tqdm(range(epochs)):
        for batch_idx, (data, target) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target) + l1_reg * l1_loss(model)
            loss.backward()
            optimizer.step()
    # for epoch in range(epochs):
    #     optimizer.zero_grad()
    #     y_pred = model(X)
    #     loss = loss_fn(y_pred, y) + l1_reg * l1_loss(model)
    #     loss.backward()
    #     optimizer.step()
    return model

def test_torch_model(model, X, y, metric='accuracy'):
    """
        Test PyTorch model on given data
    """
    # device = model_utils.device
    model = model.to(device)
    X = ch.tensor(X, dtype=ch.float32).to(device)
    y = ch.tensor(np.argmax(y, axis=1), dtype=ch.long).to(device) # Convert multi-target tensor to class labels
    y_pred = model(X)
    print(y_pred)
    # test_loss = nn.CrossEntropyLoss()(y_pred, y).item()
    if metric == 'accuracy':
        test_acc = (y_pred.argmax(1) == y).type(ch.float32).mean().item()
    elif metric == 'auc':
        test_acc = roc_auc_score(y.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
    # return test_loss, test_acc
    return test_acc

# def CSMIA_attack(model, X_test, y_test, meta):
#     dfs = [X_test.copy() for _ in range(len(meta["sensitive_values"]))]
#     sensitive_columns = [f'{meta["sensitive_column"]}_{i}' for i in range(len(meta["sensitive_values"]))]
#     for i in range(len(dfs)):
#         dfs[i][sensitive_columns] = 0
#         dfs[i][f'{meta["sensitive_column"]}_{i}'] = 1
    
#     y_preds = [np.argmax(model.predict_proba(df), axis=1) for df in dfs]
#     y_preds = np.array(y_preds).T
#     y_matched 

def LOMIA_attack(model, X_test, y_test, meta):
    attack_dataset = []
    lomia_indices = []
    correct_indices = []
    for i in tqdm(range(len(X_test))):
        # Get the predicted label and true label for this record
        #pred_label = predicted_labels[i]
        # true_label = y_test.iloc[i]
        # true_label = y_enc.transform([y_test.iloc[i]])[0]
        true_label = y_test[i]
        
        # Check if the predicted label matches the true label for only one possible value of the sensitive attribute
        num_matches = 0
        matched_value = None
        # sensitive_values = ["Married", "Single"]
        sensitive_values = meta["sensitive_values"]
        sensitive_attr = meta["sensitive_column"]
        predictions = []
        for sensitive_value in sensitive_values:
            record = X_test.iloc[i:i+1].copy()
            # record[sensitive_attr + "_" + sensitive_value] = 1
            record[f'{sensitive_attr}_{sensitive_value}'] = 1

            for other_value in sensitive_values:
                if other_value != sensitive_value:
                    # record[sensitive_attr + "_" + other_value] = 0
                    record[f'{sensitive_attr}_{other_value}'] = 0
            
            # Check if the predicted label matches the true label for this sensitive value
            # if clf.predict([record])[0] == true_label:
            # prediction = np.argmax(model.predict(record.to_numpy().reshape(1, -1))[0])
            prediction = np.argmax(model.predict(record))
            # print(prediction)
            # if model.predict(record.to_numpy().reshape(1, -1))[0] == true_label:
            if prediction == true_label:
                num_matches += 1
                matched_value = sensitive_value
                
        # If there is only one match, label the record with the matched value
        if num_matches == 1:
            record = X_test.iloc[i:i+1].copy()
            # record[sensitive_attr + "_" + matched_value] = 1
            if record[f'{sensitive_attr}_{matched_value}'].to_numpy() == 1:
                correct_indices.append(i)
            record[f'{sensitive_attr}_{matched_value}'] = 1

            for other_value in sensitive_values:
                if other_value != matched_value:
                    # record[sensitive_attr + "_" + other_value] = 0
                    record[f'{sensitive_attr}_{other_value}'] = 0
            
            # record[data_dict['y_column']] = (true_label == data_dict['y_pos'])
            record[meta['y_column']] = true_label
            attack_dataset.append(record)
            lomia_indices.append(i)
            
    return attack_dataset, lomia_indices, correct_indices


def predict_proba_for_mitiagtor(mitigator, X):
    pred = pd.DataFrame()
    for t in range(len(mitigator._hs)):
        if mitigator.weights_[t] == 0:
            pred[t] = np.zeros(len(X))
        else:
            pred[t] = mitigator._hs[t]._classifier.predict_proba(X).max(axis=1)
    
    if isinstance(mitigator.constraints, ClassificationMoment):
        positive_probs = pred[mitigator.weights_.index].dot(mitigator.weights_).to_frame()
        return np.concatenate((1 - positive_probs, positive_probs), axis=1)
    else:
        return pred

def CSMIA_attack(model, X_test, y_test, meta):
    dfs = [X_test.copy() for _ in range(len(meta["sensitive_values"]))]
    sensitive_columns = [f'{meta["sensitive_column"]}_{i}' for i in range(len(meta["sensitive_values"]))]
    for i in range(len(dfs)):
        dfs[i][sensitive_columns] = 0
        dfs[i][f'{meta["sensitive_column"]}_{i}'] = 1
    
    if isinstance(model, ExponentiatedGradient):
        y_confs = np.array([np.max(predict_proba_for_mitiagtor(model, df), axis=1) for df in dfs]).T
        y_preds = [np.argmax(model._pmf_predict(df), axis=1)==y_test.ravel() for df in dfs]
    elif isinstance(model, DecisionTreeClassifier):
        y_confs = np.array([np.max(model.predict_proba(df)[1], axis=1) for df in dfs]).T
        y_preds = [np.argmax(model.predict_proba(df)[1], axis=1)==y_test.ravel() for df in dfs]
    else:
        y_confs = np.array([np.max(model.predict_proba(df), axis=1) for df in dfs]).T
        y_preds = [np.argmax(model.predict_proba(df), axis=1)==y_test.ravel() for df in dfs]
    y_preds = np.array(y_preds).T
    case_1_indices = (y_preds.sum(axis=1) == 1)
    case_2_indices = (y_preds.sum(axis=1) > 1)
    case_3_indices = (y_preds.sum(axis=1) == 0)

    eq_conf_indices = np.argwhere(y_confs[:, 0] == y_confs[:, 1]).ravel()
    # randomly add eps to one of the confidences for the records with equal confidences
    y_confs[eq_conf_indices, np.random.randint(0, 2, len(eq_conf_indices))] += 1e-6

    sens_pred = np.zeros(y_preds.shape[0])
    sens_pred[case_1_indices] = np.argmax(y_preds[case_1_indices], axis=1)
    sens_pred[case_2_indices] = np.argmax(y_confs[case_2_indices], axis=1)
    sens_pred[case_3_indices] = np.argmin(y_confs[case_3_indices], axis=1)
    return sens_pred, {1: case_1_indices, 2: case_2_indices, 3: case_3_indices}

def get_CSMIA_case_by_case_results(clf, X_train, y_tr, ds, subgroup_col_name, metric='precision', attack_fun=None, **kwargs):
    if attack_fun is None:
        attack_fun = CSMIA_attack
    if kwargs:
        sens_pred, case_indices = attack_fun(clf, X_train, y_tr, ds.ds.meta, **kwargs)
    else:
        sens_pred, case_indices = attack_fun(clf, X_train, y_tr, ds.ds.meta)
    sensitive_col_name = f'{ds.ds.meta["sensitive_column"]}_1'
    correct_indices = (sens_pred == X_train[[sensitive_col_name]].to_numpy().ravel())

    # subgroup_csmia_case_dict = {
    #     i: X_train.iloc[np.argwhere(case_indices[i]).ravel()][f'{subgroup_col_name}_1'].value_counts() for i in range(1, 4)
    # }

    subgroup_csmia_case_indices_by_subgroup_dict = {
        i: { j: np.intersect1d(np.argwhere(case_indices[i]).ravel(), np.argwhere(X_train[f'{subgroup_col_name}_1'].to_numpy().ravel() == j).ravel()) for j in [1, 0] } for i in range(1, 4)
    }

    subgroup_csmia_case_indices_by_subgroup_dict['All Cases'] = { j: np.argwhere(X_train[f'{subgroup_col_name}_1'].to_numpy().ravel() == j).ravel() for j in [1, 0] }

    def fun(metric):
        if metric.__name__ in ['precision_score', 'recall_score', 'f1_score']:
            return lambda x: round(100 * metric(x[0], x[1], pos_label=0), 4)
        else:
            return lambda x: round(100 * metric(x[0], x[1]), 4)
    
    def fun2(x):
        tp, fn, fp, tn = confusion_matrix(x[0], x[1]).ravel()
        return f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}"
    
    def false_positive_rate(x):
        tp, fn, fp, tn = confusion_matrix(x[0], x[1]).ravel()
        return round(100 * fp / (fp + tn), 4)

    eval_func = { 
        'precision': fun(precision_score),
        'recall': fun(recall_score),
        'f1': fun(f1_score),
        'accuracy': fun(accuracy_score),
        'fpr': false_positive_rate,
        # 'confusion_matrix': lambda x: f"TP: {confusion_matrix(x[0], x[1], labels=labels)[0, 0]}, FP: {confusion_matrix(x[0], x[1], labels=labels)[0, 1]}, FN: {confusion_matrix(x[0], x[1], labels=labels)[1, 0]}, TN: {confusion_matrix(x[0], x[1], labels=labels)[1, 1]}",
        'confusion_matrix': fun2,
        'mcc': fun(matthews_corrcoef),
        'gmean': fun(geometric_mean_score),
    }[metric]

    perf_dict = {
        i: { j: eval_func((X_train.loc[subgroup_csmia_case_indices_by_subgroup_dict[i][j], sensitive_col_name], sens_pred[subgroup_csmia_case_indices_by_subgroup_dict[i][j]])) for j in [1, 0] } for i in [1, 2, 3, 'All Cases']
    }

    overall_perf_by_cases_dict = {
        i: eval_func((X_train.loc[case_indices[i]].loc[:, sensitive_col_name], sens_pred[case_indices[i]])) for i in [1, 2, 3]
    }
    overall_perf_by_cases_dict['All Cases'] = eval_func((X_train.loc[:, sensitive_col_name], sens_pred))

    temp_dict = {
        f'Case {i}': { j: f'{subgroup_csmia_case_indices_by_subgroup_dict[i][j].shape[0]} ({perf_dict[i][j]})' for j in [1, 0] } for i in [1, 2, 3, 'All Cases']
    }

    for i in [1, 2, 3, 'All Cases']:
        temp_dict[f'Case {i}']['Overall'] = overall_perf_by_cases_dict[i]

    # print(temp_dict)

    # subgroup_csmia_case_correct_dict = {
    #     i: X_train.iloc[np.intersect1d(np.argwhere(case_indices[i]).ravel(), np.argwhere(correct_indices).ravel())][f'{subgroup_col_name}_1'].value_counts() for i in range(1, 4)
    # }

    # temp_dict = {
    #     f'Case {i}': { j: f'{subgroup_csmia_case_dict[i][j]} ({round(100 * subgroup_csmia_case_correct_dict[i][j] / subgroup_csmia_case_dict[i][j], 2)})' for j in [1, 0] } for i in range(1, 4)
    # }
    # temp_dict['All Cases'] = { j: f'{subgroup_csmia_case_dict[1][j] + subgroup_csmia_case_dict[2][j] + subgroup_csmia_case_dict[3][j]} ({round(100 * (subgroup_csmia_case_correct_dict[1][j] + subgroup_csmia_case_correct_dict[2][j] + subgroup_csmia_case_correct_dict[3][j]) / (subgroup_csmia_case_dict[1][j] + subgroup_csmia_case_dict[2][j] + subgroup_csmia_case_dict[3][j]), 2)})' for j in [1, 0] }

    temp_df = pd.DataFrame.from_dict(temp_dict, orient='index')
    # temp_df['Overall'] = overall_perf_by_cases_dict
    return temp_df

