import data_utils
from model_utils import get_model
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import re

def false_positive_rate(x, y):
    tp, fn, fp, tn = confusion_matrix(x, y).ravel()
    return round(fp / (fp + tn), 4)

eval_func = {
    'precision': precision_score,
    'recall': recall_score,
    'f1': f1_score,
    'accuracy': accuracy_score,
    'fpr': false_positive_rate
}

metrics = ['accuracy', 'precision', 'recall', 'fpr', 'f1']

def get_perf(input_string):
    # Use regular expression to extract the number inside parenthesis
    match = re.search(r'\(([\d.]+)\)', input_string)

    if match:
        extracted_number = match.group(1)
        return(float(extracted_number))
    else:
        return(np.nan)

class MIAExperiment:
    def __init__(self, *args, **kwargs):
        self.sampling_condition_dict = kwargs.get('sampling_condition_dict', None)
        self.sensitive_column = kwargs.get('sensitive_column', 'MAR')

        for key, value in kwargs.items():
            setattr(self, key, value)

        if not hasattr(self, 'name'):
            self.name = 'Census19'
        if not hasattr(self, 'random_state'):
            self.random_state = 42
        self.ds = data_utils.CensusWrapper(
                    filter_prop="none", ratio=float(0.5), split="all", name=self.name, sampling_condition_dict=self.sampling_condition_dict, sensitive_column=self.sensitive_column,
                    additional_meta=None, random_state=self.random_state)
        # print(self.sampling_condition_dict)
        # print(self.ds.ds.filenameroot)
        (self.x_tr, self.y_tr), (self.x_te, self.y_te), self.cols = self.ds.load_data()
        self.X_train = pd.DataFrame(self.x_tr, columns=self.cols)
        self.X_test = pd.DataFrame(self.x_te, columns=self.cols)
        self.y_tr_onehot = self.ds.ds.y_enc.transform(self.y_tr).toarray()
        self.y_te_onehot = self.ds.ds.y_enc.transform(self.y_te).toarray()

        sensitive_columns = [f'{self.ds.ds.meta["sensitive_column"]}_{i}' for i in self.ds.ds.meta["sensitive_values"]]
        self.sens_val_ground_truth = self.X_train[sensitive_columns].idxmax(axis=1).str.replace(f'{self.ds.ds.meta["sensitive_column"]}_', '')
        self.sens_val_ground_truth = self.sens_val_ground_truth.astype(self.ds.ds.original_df[self.ds.ds.meta["sensitive_column"]].dtype)
        self.sens_val_ground_truth = np.array([{x: i for i, x in enumerate(self.ds.ds.meta["sensitive_values"])}[val] for val in self.sens_val_ground_truth])
        # self.sens_vals_ground_truth = self.X_train[f'{self.sensitive_column}_{self.ds.ds.meta["sensitive_positive"]}']

        if not hasattr(self, 'hidden_layer_sizes'):
            self.hidden_layer_sizes = None

    def __str__(self):
        return self.ds.ds.filenameroot
    
    def __repr__(self):
        return self.ds.ds.filenameroot
    
    def get_value_count_report(self):
        df = self.ds.ds.original_df
        df = df[df['is_train'] == 1]
        subgroup_values = df[self.subgroup_column].unique().tolist()
        for value in subgroup_values:
            print(f"Subgroup: {value}")
            # print(df[df[self.subgroup_column] == value].columns)
            # print(df[df[self.subgroup_column] == value][[self.sensitive_column, self.y_column]])
            new_df = df[df[self.subgroup_column] == value][[self.sensitive_column, self.y_column]]
            print(new_df.value_counts())
            # print(df[df[self.subgroup_column == value]][[self.sensitive_column, self.y_column]].corr())


    def get_mutual_information_between_sens_and_y(self):
        df = self.ds.ds.original_df
        df = df[df['is_train'] == 1]
        subgroup_values = df[self.subgroup_column].unique().tolist()
        mutual_info_dict = {}
        for value in subgroup_values:
            print(f"Subgroup: {value}")
            # All the features except y column
            X = df[df[self.subgroup_column] == value].drop([self.y_column], axis=1)
            y = df[df[self.subgroup_column] == value][[self.y_column]]
            # print(mutual_info_classif(X, y, discrete_features=True))
            mutual_info_dict[value] = mutual_info_classif(X, y, discrete_features=True)
        return mutual_info_dict

    def get_base_model(self):
        if self.hidden_layer_sizes is None:
            return model_utils.get_model(max_iter=500)
        else:
            return model_utils.get_model(max_iter=500, hidden_layer_sizes=self.hidden_layer_sizes)

    def score(self, sens_true, sens_pred, metric='accuracy'):
        pos_label = self.ds.ds.meta['sensitive_values'].index(self.ds.ds.meta['sensitive_positive'])

        def false_positive_rate(x, y, pos_label=pos_label):
            tp, fn, fp, tn = confusion_matrix(x, y).ravel()
            return round(fp / (fp + tn), 4)

        def accuracy(x, y, pos_label=None):
            return accuracy_score(x, y)

        eval_func = {
            'precision': precision_score,
            'recall': recall_score,
            'f1': f1_score,
            'accuracy': accuracy,
            'fpr': false_positive_rate
        }

        return round(100 * eval_func[metric](sens_true, sens_pred, pos_label=pos_label), 2)