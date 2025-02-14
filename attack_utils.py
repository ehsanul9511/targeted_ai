from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fairlearn.reductions import ExponentiatedGradient
from fairlearn.reductions._moments import ClassificationMoment
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, matthews_corrcoef, confusion_matrix
from imblearn.metrics import geometric_mean_score
import model_utils
from whitebox_attack import neuron_output, make_neuron_output_data, wb_corr_attacks
from disparate_vulnerability_utils import get_disparity_by_subgroup, get_imputation_prediction

def LOMIA_attack_old(model, X_test, y_test, meta):
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

# def Yeom_attack(experiment, model, X_test, y_test, meta):
def imputation_attack(experiment, X_target, y_target, X_aux, y_aux, meta):
    X_attack_aux = X_aux.copy()
    y_oh_columns = [f'{meta["y_column"]}_{i}' for i in meta["y_values"]]
    X_attack_aux[y_oh_columns] = experiment.ds.ds.y_enc.transform(y_aux.ravel().reshape(-1, 1)).toarray()
    sensitive_columns = [f'{meta["sensitive_column"]}_{i}' for i in meta["sensitive_values"]]
    X_attack_aux.drop(sensitive_columns, axis=1, inplace=True)
    # y_attack_aux = X_aux[sensitive_columns].idxmax(axis=1).apply(lambda x: int(x.split('_')[-1]))
    y_attack_aux = X_aux[sensitive_columns]

    inv_clf = model_utils.get_model(max_iter=100)
    inv_clf.fit(X_attack_aux, y_attack_aux)

    X_attack_target = X_target.copy()
    X_attack_target.drop(sensitive_columns, axis=1, inplace=True)
    # y_attack_target = X_target[sensitive_columns]
    X_attack_target[y_oh_columns] = experiment.ds.ds.y_enc.transform(y_target.ravel().reshape(-1, 1)).toarray()

    return np.argmax(inv_clf.predict(X_attack_target), axis=1)

def whitebox_neuron_attack(experiment, clf, X_target, y_target, X_aux, y_aux, meta):
    X_attack_aux = X_aux.copy()
    y_oh_columns = [f'{meta["y_column"]}_{i}' for i in meta["y_values"]]
    X_attack_aux[y_oh_columns] = experiment.ds.ds.y_enc.transform(y_aux.ravel().reshape(-1, 1)).toarray()
    sensitive_columns = [f'{meta["sensitive_column"]}_{i}' for i in meta["sensitive_values"]]
    X_attack_aux.drop(sensitive_columns, axis=1, inplace=True)
    y_attack_aux = X_aux[sensitive_columns].idxmax(axis=1).apply(lambda x: int(x.split('_')[-1]))
    X_attack_aux_neuron = make_neuron_output_data(experiment.ds, X_attack_aux, clf, experiment.ds.ds.y_columns)

    top_10_corr_neurons_model = wb_corr_attacks(X_attack_aux_neuron, y_attack_aux)

    X_attack_target = X_target.copy()
    X_attack_target.drop(sensitive_columns, axis=1, inplace=True)
    X_attack_target[y_oh_columns] = experiment.ds.ds.y_enc.transform(y_target.ravel().reshape(-1, 1)).toarray()
    X_attack_target_neuron = make_neuron_output_data(experiment.ds, X_attack_target, clf, experiment.ds.ds.y_columns)

    preds = top_10_corr_neurons_model(torch.from_numpy(X_attack_target_neuron.to_numpy()).float()).detach().numpy()
    preds = np.where(preds > top_10_corr_neurons_model.threshold, 1, 0)
    return preds
    

def imputation_attack_old(experiment, subgroup_column='SEX', aux_df_onehot=None, pred_on_train=True, is_train=0):
    original_df_onehot = experiment.ds.ds.df.copy()

    original_df_onehot = original_df_onehot[original_df_onehot["is_train"]==1].drop("is_train", axis=1).reset_index(drop=True)

    X_attack_orig, y_sens_orig = experiment.ds.ds.get_attack_df(original_df_onehot)
    X_attack_orig = X_attack_orig.astype(float)
    y_sens_orig_onehot = experiment.ds.ds.sensitive_enc.transform(y_sens_orig.to_numpy().ravel().reshape(-1, 1)).toarray()

    if aux_df_onehot is None:
        aux_df_onehot = experiment.ds.ds.df.copy()
        aux_df_onehot = aux_df_onehot[aux_df_onehot["is_train"]==1].drop("is_train", axis=1).reset_index(drop=True)
    else:
        aux_df_onehot = aux_df_onehot[aux_df_onehot["is_train"]==is_train].drop("is_train", axis=1).reset_index(drop=True)


    X_attack_aux, y_sens_aux = experiment.ds.ds.get_attack_df(aux_df_onehot)
    # aux_samp_indices = np.random.choice(X_attack_aux.index, aux_count, False)
    # X_attack_aux, y_sens_aux = X_attack_aux.iloc[aux_samp_indices], y_sens_aux[aux_samp_indices]
    X_attack_aux = X_attack_aux.astype(float)
    y_sens_aux_onehot = experiment.ds.ds.sensitive_enc.transform(y_sens_aux.to_numpy().ravel().reshape(-1, 1)).toarray()

    inv_clf = model_utils.get_model(max_iter=100)
    inv_clf.fit(X_attack_aux, y_sens_aux_onehot)

    # y_pred = np.argmax(inv_clf.predict(X_attack_orig), axis=1)

    # result_dict = get_disparity_by_subgroup(attack_type='INV', ds=experiment.ds, subgroup_columns=['SEX'], X_att_query=X_attack_orig, y_att_query=y_sens_orig, metric='accuracy', clf = inv_clf)
    # experiment.inv_clf = inv_clf
    return np.argmax(inv_clf.predict(X_attack_orig), axis=1)
    if pred_on_train:
        return get_imputation_prediction(ds=experiment.ds, subgroup_columns={subgroup_column}, X_att_query=X_attack_orig, y_att_query=y_sens_orig, clf = inv_clf)
    else:
        return get_imputation_prediction(ds=experiment.ds, subgroup_columns={subgroup_column}, X_att_query=X_attack_aux, y_att_query=y_sens_aux, clf = inv_clf)

    result_dict = {}
    for metric in metrics:
        temp_dict = get_disparity_by_subgroup(attack_type='INV', ds=experiment.ds, subgroup_columns={subgroup_column}, X_att_query=X_attack_orig, y_att_query=y_sens_orig, metric=metric, clf = inv_clf)[subgroup_column]
        result_dict[metric] = temp_dict
        # result_dict[metric] = temp_dict
    return result_dict

def whitebox_neuron_attack_old(experiment, clf, subgroup_column="SEX", metrics=['accuracy', 'precision', 'recall', 'fpr', 'f1', 'auc'], aux_df=None):
    original_df_onehot = experiment.ds.ds.df.copy()

    original_df_onehot = original_df_onehot[original_df_onehot["is_train"]==0].drop("is_train", axis=1)

    X_attack_orig, y_sens_orig = experiment.ds.ds.get_attack_df(original_df_onehot)
    X_attack_orig = X_attack_orig.astype(float)

    if aux_df is not None:
        aux_df_ = aux_df[aux_df["is_train"]==0].drop("is_train", axis=1)
        X_attack_ , y_sens_ = experiment.ds.ds.get_attack_df(aux_df_)
        X_attack_ = X_attack_.astype(float)
    else:
        X_attack_ , y_sens_ = experiment.ds.ds.get_attack_df(original_df_onehot)
        X_attack_ = X_attack_.astype(float)

    x_tr, x_te, y_tr, y_te = train_test_split(X_attack_, y_sens_, test_size=0.9, random_state=42)
    x_n_tr = make_neuron_output_data(experiment.ds, x_tr, clf, experiment.ds.ds.y_columns)
    x_n_te = make_neuron_output_data(experiment.ds, x_te, clf, experiment.ds.ds.y_columns)

    top_10_corr_neurons_model = wb_corr_attacks(x_n_tr, y_tr)

    X_n_attack_orig = make_neuron_output_data(experiment.ds, X_attack_orig, clf, experiment.ds.ds.y_columns)

    y_preds = top_10_corr_neurons_model(torch.from_numpy(X_n_attack_orig.to_numpy()).float()).detach().numpy()
    y_preds = np.where(y_preds > top_10_corr_neurons_model.threshold, 1, 0)

    result_dict = {}
    for metric in metrics:
        temp_dict = get_disparity_by_subgroup(attack_type='NEUR_IMP', ds=experiment.ds, subgroup_columns=[subgroup_column], X_att_query=X_attack_orig, y_att_query=y_sens_orig, metric=metric, clf=top_10_corr_neurons_model, df=X_attack_orig, MLP=clf)[subgroup_column]
        result_dict[metric] = temp_dict['original']


    # result_dict = {metric: get_disparity_by_subgroup(attack_type='NEUR_IMP', ds=experiment.ds, subgroup_columns=[subgroup_column], X_att_query=X_attack_orig, y_att_query=y_sens_orig, metric=metric, clf=top_10_corr_neurons_model, df=X_attack_orig, MLP=clf)[subgroup_column] for metric in metrics}
    return result_dict

def LOMIA_attack(experiment, model, X_test, y_test, meta, indices=None):
    sens_pred, case_indices = CSMIA_attack(model, X_test, y_test, meta)
    case_1_indices = case_indices[1]

    if case_1_indices.sum() == 0:
        # random attack
        return np.random.choice(meta["sensitive_values"], len(X_test))

    # original_df_onehot = experiment.ds.ds.df.copy()
    # original_df_onehot = original_df_onehot[original_df_onehot["is_train"]==1].drop("is_train", axis=1).reset_index(drop=True)
    # X_attack_orig, y_sens_orig = experiment.ds.ds.get_attack_df(original_df_onehot)
    X_attack_orig = X_test.copy()
    y_values = meta["y_values"]
    y_column = meta["y_column"]
    y_oh_columns = [f'{y_column}_{y_value}' for y_value in y_values]
    X_attack_orig[y_oh_columns] = 0
    X_attack_orig[y_oh_columns] = experiment.ds.ds.y_enc.transform(y_test.ravel().reshape(-1, 1)).toarray()
    sensitive_columns = [f'{meta["sensitive_column"]}_{i}' for i in meta["sensitive_values"]]
    X_attack_orig.drop(sensitive_columns, axis=1, inplace=True)

    if indices is not None:
        # X_attack_orig, y_sens_orig = X_attack_orig.loc[indices].reset_index(drop=True), y_sens_orig[indices]
        X_attack_orig = X_attack_orig.loc[indices].reset_index(drop=True)

    X_attack, y_attack = X_attack_orig.iloc[case_1_indices], sens_pred[case_1_indices]
    y_attack_onehot = experiment.ds.ds.sensitive_enc.transform(y_attack.ravel().reshape(-1, 1)).toarray()

    attack_clf = model_utils.get_model(max_iter=40)
    attack_clf.fit(X_attack, y_attack_onehot)

    sens_pred_LOMIA = np.argmax(attack_clf.predict(X_attack_orig), axis=1)
    return sens_pred_LOMIA

def LOMIA_attack_old(experiment, model, X_test, y_test, meta, indices=None):
    sens_pred, case_indices = CSMIA_attack(model, X_test, y_test, meta)
    case_1_indices = case_indices[1]

    original_df_onehot = experiment.ds.ds.df.copy()
    original_df_onehot = original_df_onehot[original_df_onehot["is_train"]==1].drop("is_train", axis=1).reset_index(drop=True)
    X_attack_orig, y_sens_orig = experiment.ds.ds.get_attack_df(original_df_onehot)

    if indices is not None:
        X_attack_orig, y_sens_orig = X_attack_orig.loc[indices].reset_index(drop=True), y_sens_orig[indices]

    X_attack, y_attack = X_attack_orig.iloc[case_1_indices], sens_pred[case_1_indices]
    y_attack_onehot = experiment.ds.ds.sensitive_enc.transform(y_attack.ravel().reshape(-1, 1)).toarray()

    attack_clf = model_utils.get_model(max_iter=40)
    attack_clf.fit(X_attack, y_attack_onehot)

    sens_pred_LOMIA = np.argmax(attack_clf.predict(X_attack_orig), axis=1)
    return sens_pred_LOMIA


# def advanced_CSMIA(model, X_test, y_test, meta):
#     dfs = [X_test.copy() for _ in range(len(meta["sensitive_values"]))]
#     sensitive_columns = [f'{meta["sensitive_column"]}_{i}' for i in range(len(meta["sensitive_values"]))]
#     for i in range(len(dfs)):
#         dfs[i][sensitive_columns] = 0
#         dfs[i][f'{meta["sensitive_column"]}_{i}'] = 1
    
#     if isinstance(model, ExponentiatedGradient):
#         y_confs = np.array([np.max(predict_proba_for_mitiagtor(model, df), axis=1) for df in dfs]).T
#         y_preds = [np.argmax(model._pmf_predict(df), axis=1)==y_test.ravel() for df in dfs]
#     elif isinstance(model, DecisionTreeClassifier):
#         y_confs = np.array([np.max(model.predict_proba(df)[1], axis=1) for df in dfs]).T
#         y_preds = [np.argmax(model.predict_proba(df)[1], axis=1)==y_test.ravel() for df in dfs]
#     else:
#         y_confs = np.array([np.max(model.predict_proba(df), axis=1) for df in dfs]).T
#         y_preds = [np.argmax(model.predict_proba(df), axis=1)==y_test.ravel() for df in dfs]
#     y_preds = np.array(y_preds).T
    
#     y_test_onehot = experiment.ds.ds.y_enc.transform(y_test).toarray()
#     y_test_onehot[y_test_onehot == 0] = -1

#     y_conf_gap = 

def FJRMIA_attack(model, X_test, y_test, meta):
    dfs = [X_test.copy() for _ in range(len(meta["sensitive_values"]))]
    sensitive_columns = [f'{meta["sensitive_column"]}_{i}' for i in range(len(meta["sensitive_values"]))]
    for i in range(len(dfs)):
        dfs[i][sensitive_columns] = 0
        dfs[i][f'{meta["sensitive_column"]}_{i}'] = 1
    
    if isinstance(model, ExponentiatedGradient):
        y_pred_ = np.argmax(model._pmf_predict(X_test), axis=1)
        y_preds = [np.argmax(model._pmf_predict(df), axis=1) for df in dfs]
    elif isinstance(model, DecisionTreeClassifier):
        y_pred_ = np.argmax(model.predict_proba(X_test)[1], axis=1)
        y_preds = [np.argmax(model.predict_proba(df)[1], axis=1) for df in dfs]
    else:
        y_pred_ = np.argmax(model.predict_proba(X_test), axis=1)
        y_preds = np.array([np.argmax(model.predict_proba(df), axis=1) for df in dfs]).T

    y_test = y_test.ravel()
    mp = np.array([X_test[f'{meta["sensitive_column"]}_{i}'].value_counts()[1] for i in range(len(meta["sensitive_values"]))])
    mp = mp/mp.sum()
    cm = np.array(confusion_matrix(y_test.ravel(), y_pred_))
    cm = cm / cm.sum(axis=1, keepdims=True)
    z = np.zeros((y_test.shape[0], cm.shape[1]))

    # Populate the output array based on the conditions
    z[y_test == 0] = cm[0]
    z[y_test == 1] = cm[1]

    row_indices = np.arange(y_preds.shape[0])[:, None]
    c = z[row_indices, y_preds]

    cp = c * mp

    eq_cp_indices = np.argwhere(cp[:, 0] == cp[:, 1]).ravel()
    cp[eq_cp_indices, np.random.randint(0, 2, len(eq_cp_indices))] += 1e-6

    sens_pred_FJRMIA = np.argmax(cp, axis=1)

    return sens_pred_FJRMIA



def CSMIA_attack(model, X_test, y_test, meta):
    dfs = [X_test.copy() for _ in range(len(meta["sensitive_values"]))]
    sensitive_values = meta["sensitive_values"]
    sensitive_columns = [f'{meta["sensitive_column"]}_{i}' for i in sensitive_values]
    for i in range(len(dfs)):
        dfs[i][sensitive_columns] = 0
        dfs[i][f'{meta["sensitive_column"]}_{sensitive_values[i]}'] = 1

    # print(dfs[i].columns)
    
    if isinstance(model, nn.Module):
        for i in range(len(dfs)):
            dfs[i] = torch.tensor(dfs[i].values).float().to('mps')
        y_confs = np.array([np.max(model.predict_proba(df).cpu().detach().numpy(), axis=1) for df in dfs]).T
        y_preds = [np.argmax(model.predict_proba(df).cpu().detach().numpy(), axis=1)==y_test.ravel() for df in dfs]
    elif isinstance(model, ExponentiatedGradient):
        y_confs = np.array([np.max(predict_proba_for_mitiagtor(model, df), axis=1) for df in dfs]).T
        y_preds = [np.argmax(model._pmf_predict(df), axis=1)==y_test.ravel() for df in dfs]
    elif isinstance(model, DecisionTreeClassifier) or isinstance(model, RandomForestClassifier):
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

def get_CSMIA_case_by_case_results_old(clf, X_train, y_tr, ds, subgroup_col_name, metric='precision', attack_fun=None, **kwargs):
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


def get_LOMIA_results(experiment, model, X_test, y_test, meta, sensitive_col_name, subgroup_col_name, subgroup_vals, metric='accuracy'):
    sens_pred = LOMIA_attack(experiment, model, X_test, y_test, meta)

    def fun(metric):
        if metric.__name__ in ['precision_score', 'recall_score', 'f1_score']:
            return lambda x: round(100 * metric(x[0], x[1], pos_label=1), 4)
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
        # 'mcc': fun(matthews_corrcoef),
        # 'gmean': fun(geometric_mean_score),
    }[metric]

    subgroup_indices_dict = {i: np.argwhere(X_test[f'{subgroup_col_name}_{i}'].to_numpy().ravel() == 1).ravel() for i in subgroup_vals}

    # print(subgroup_indices_dict)

    perf_dict = {i: eval_func((X_test.loc[subgroup_indices_dict[i], sensitive_col_name], sens_pred[subgroup_indices_dict[i]])) for i in subgroup_vals}

    return perf_dict



def get_CSMIA_case_by_case_results(clf, X_train, y_tr, ds, subgroup_col_name, metric='precision', sensitive_col_name='marital_Single', feature_dim_version=False, subgroup_vals=[1, 0]):
    if sensitive_col_name is None:
        sensitive_col_name = f'{ds.ds.meta["sensitive_column"]}_1'
        sens_pred, case_indices = CSMIA_attack(clf, X_train, y_tr, ds.ds.meta)
    else:
        sens_pred, case_indices = CSMIA_attack(clf, X_train, y_tr, ds.ds.meta)
    correct_indices = (sens_pred == X_train[[sensitive_col_name]].to_numpy().ravel())

    # subgroup_csmia_case_dict = {
    #     i: X_train.iloc[np.argwhere(case_indices[i]).ravel()][f'{subgroup_col_name}_1'].value_counts() for i in range(1, 4)
    # }

    subgroup_csmia_case_indices_by_subgroup_dict = {
        i: { j: np.intersect1d(np.argwhere(case_indices[i]).ravel(), np.argwhere(X_train[f'{subgroup_col_name}_{j}'].to_numpy().ravel() == 1).ravel()) for j in subgroup_vals } for i in range(1, 4)
    }

    subgroup_csmia_case_indices_by_subgroup_dict['All Cases'] = { j: np.argwhere(X_train[f'{subgroup_col_name}_{j}'].to_numpy().ravel() == 1).ravel() for j in subgroup_vals }

    def fun(metric):
        if metric.__name__ in ['precision_score', 'recall_score', 'f1_score']:
            return lambda x: round(100 * metric(x[0], x[1], pos_label=1), 4)
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
        # 'mcc': fun(matthews_corrcoef),
        # 'gmean': fun(geometric_mean_score),
    }[metric]

    perf_dict = {
        i: { j: eval_func((X_train.loc[subgroup_csmia_case_indices_by_subgroup_dict[i][j], sensitive_col_name], sens_pred[subgroup_csmia_case_indices_by_subgroup_dict[i][j]])) for j in subgroup_vals } for i in [1, 2, 3, 'All Cases']
    }

    overall_perf_by_cases_dict = {
        i: eval_func((X_train.loc[case_indices[i]].loc[:, sensitive_col_name], sens_pred[case_indices[i]])) for i in [1, 2, 3]
    }
    overall_perf_by_cases_dict['All Cases'] = eval_func((X_train.loc[:, sensitive_col_name], sens_pred))

    temp_dict = {
        f'Case {i}': { j: f'{subgroup_csmia_case_indices_by_subgroup_dict[i][j].shape[0]} ({perf_dict[i][j]})' for j in subgroup_vals } for i in [1, 2, 3, 'All Cases']
    }

    for i in [1, 2, 3, 'All Cases']:
        temp_dict[f'Case {i}']['Overall'] = overall_perf_by_cases_dict[i]


    # subgroup_csmia_case_correct_dict = {
    #     i: X_train.iloc[np.intersect1d(np.argwhere(case_indices[i]).ravel(), np.argwhere(correct_indices).ravel())][f'{subgroup_col_name}_1'].value_counts() for i in range(1, 4)
    # }

    # temp_dict = {
    #     f'Case {i}': { j: f'{subgroup_csmia_case_dict[i][j]} ({round(100 * subgroup_csmia_case_correct_dict[i][j] / subgroup_csmia_case_dict[i][j], 2)})' for j in [1, 0] } for i in range(1, 4)
    # }
    # temp_dict['All Cases'] = { j: f'{subgroup_csmia_case_dict[1][j] + subgroup_csmia_case_dict[2][j] + subgroup_csmia_case_dict[3][j]} ({round(100 * (subgroup_csmia_case_correct_dict[1][j] + subgroup_csmia_case_correct_dict[2][j] + subgroup_csmia_case_correct_dict[3][j]) / (subgroup_csmia_case_dict[1][j] + subgroup_csmia_case_dict[2][j] + subgroup_csmia_case_dict[3][j]), 2)})' for j in [1, 0] }

    temp_df = pd.DataFrame.from_dict(temp_dict, orient='index')

    return temp_df