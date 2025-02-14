import numpy as np
from fairlearn.reductions import ExponentiatedGradient
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

def get_indices_by_group_condition(X, conditions):
    X_new = X.copy()
    if len(conditions.keys()) == 0:
        return X_new.index.to_numpy().astype(int)
    for key, value in conditions.items():
        if isinstance(value, list):
            # If value is a list, filter for any of the conditions in the list
            condition = X_new[[f"{key}_{v}" for v in value]].sum(axis=1) > 0
            X_new = X_new[condition]
        else:
            # If value is a single item, filter for that specific condition
            X_new = X_new.loc[X_new[f"{key}_{value}"] == 1]
    return X_new.index

def get_corr_btn_sens_and_out_per_subgroup(experiment, X, y, conditions):
    sensitive_column = experiment.ds.ds.meta["sensitive_column"]
    y_column = experiment.ds.ds.meta["y_column"]
    indices = get_indices_by_group_condition(X, conditions)
    if len(indices) == 0:
        return np.nan
    X_new = X.loc[indices].copy()
    X_new[sensitive_column] = X_new[[f'{sensitive_column}_{val}' for val in experiment.ds.ds.meta["sensitive_values"]]].idxmax(axis=1).str.replace(f'{sensitive_column}_', '')
    X_new[sensitive_column] = X_new[sensitive_column].astype(experiment.ds.ds.original_df[sensitive_column].dtype)
    X_new[sensitive_column] = X_new[sensitive_column].map({val: i for i, val in enumerate(experiment.ds.ds.meta["sensitive_values"])})
    X_new[y_column] = y[indices]
    return X_new[[sensitive_column, y_column]].corr().iloc[0, 1]
    sensitive_positive = experiment.ds.ds.meta["sensitive_positive"]
    X_new = X.loc[indices]
    y_new = y[indices]
    X_new = X_new[[f'{sensitive_column}_{sensitive_positive}']].to_numpy().ravel()
    y_new = y_new.ravel()
    return np.corrcoef(X_new, y_new)[0, 1]

def get_mutual_info_btn_sens_and_out_per_subgroup(experiment, X, y, conditions):
    sensitive_column = experiment.ds.ds.meta["sensitive_column"]
    indices = get_indices_by_group_condition(X, conditions)
    X_new = X.loc[indices]
    y_new = y[indices]
    X_new = X_new[[f'{sensitive_column}_1']]
    y_new = y_new.ravel()
    return mutual_info_classif(X_new, y_new, discrete_features=True)

def get_confidence_array(experiment, X, y, model, get_conf_fun=None):
    sensitive_column = experiment.ds.ds.meta["sensitive_column"]
    sensitive_values = experiment.ds.ds.meta["sensitive_values"]
    sensitive_columns_onehot = [f'{sensitive_column}_{i}' for i in sensitive_values]
    Xs = [X.copy() for _ in range(len(sensitive_values))]
    for i in range(len(Xs)):
        Xs[i][sensitive_columns_onehot] = 0
        Xs[i][f'{sensitive_column}_{sensitive_values[i]}'] = 1

    if get_conf_fun is not None:
        y_confs = np.array([get_conf_fun(model, X, y) for X in Xs]).T
    elif isinstance(model, ExponentiatedGradient):
        y_confs = np.array([np.max(predict_proba_for_mitiagtor(model, X), axis=1) for X in Xs]).T
    elif isinstance(model, DecisionTreeClassifier) or isinstance(model, RandomForestClassifier):
        y_confs = np.array([np.max(model.predict_proba(X)[1], axis=1) for X in Xs]).T
    else:
        y_confs = np.array([np.max(model.predict_proba(X), axis=1) for X in Xs]).T

    return y_confs


def draw_confidence_array_scatter(experiment, confidence_array, y, num_points=100, style_fun=None):
    y_values = np.array(experiment.ds.ds.meta["y_values"]).astype(int)
    indices_by_y_values = {y_value: np.where(y.ravel() == y_value)[0] for y_value in y_values}
    colors = ['grey', 'black']
    markers = ['x', 'o']
    scatter_kws_list = [
        dict(s=18),
        dict(s=12, facecolor='none', edgecolor='black')
    ]
    y_names = ['Records with negative output', 'Records with positive output']
    for y_value in [1, 0]:
        sns.regplot(x = confidence_array[indices_by_y_values[y_value], 1][:num_points], y = confidence_array[indices_by_y_values[y_value], 0][:num_points], marker=markers.pop(), color=colors.pop(), line_kws=dict(linewidth=0.5), scatter_kws=scatter_kws_list.pop(), label=y_names[y_value])

    plt.ylim(0.5, 1)
    plt.xlabel('Confidence Score when queried with sensitive attribute is set to positive')
    plt.ylabel('Conf. Score when queried with sensitive attribute is set to negative')
    plt.legend()
    if style_fun is not None:
        style_fun(ax)
    plt.show()


def draw_jointplot(experiment, confidence_array, y):
    y_values = np.array(experiment.ds.ds.meta["y_values"]).astype(int)
    indices_by_y_values = {y_value: np.where(y.ravel() == y_value)[0] for y_value in y_values}
    colors = ['Reds', 'Blues']
    y_names = ['Records with negative output', 'Records with positive output']

    kde_kws = {'lw': 0.5, 'alpha': 0.5}
    vmax=10
    hex_alphas=[0.5, 0.5]

    plt.figure(figsize=(5, 6))

    grid = sns.JointGrid(x=confidence_array[indices_by_y_values[1], 1][:], y=confidence_array[indices_by_y_values[1], 0][:])

    blue_cmap = LinearSegmentedColormap.from_list('blue_cmap', [(0, 0, 1, 0.5), (0, 0, 1, 1)])
    red_cmap = LinearSegmentedColormap.from_list('red_cmap', [(1, 0, 0, 0.5), (1, 0, 0, 1)])

    # Plot the first hexbin
    hb1 = grid.plot_joint(plt.hexbin, gridsize=50, cmap='Blues', mincnt=1, vmin=1, vmax=vmax, alpha=hex_alphas[0], edgecolor=None)
    
    # Plot the second hexbin on the same grid
    hb2 = grid.ax_joint.hexbin(confidence_array[indices_by_y_values[0], 1][:], confidence_array[indices_by_y_values[0], 0][:], gridsize=50, cmap='Reds', mincnt=1, vmin=1, vmax=vmax, alpha=hex_alphas[1], edgecolor=None)

    n_bins=50
    bins_x = np.linspace(0.5, 1, n_bins+1)
    bins_y = np.linspace(0.5, 1, n_bins+1)

    color_code_1="#8397b4"
    color_code_2="#b27f85"


    # sns.histplot(confidence_array[indices_by_y_values[1], 1][:], kde=True, color=(0, 0, 1, 0.1), ax=grid.ax_marg_x, bins=bins_x, fill=True, alpha=0.5, line_kws=kde_kws)
    h1 = sns.histplot(y=confidence_array[indices_by_y_values[1], 0][:], kde=True, color=color_code_1, ax=grid.ax_marg_y, bins=bins_y, fill=True, alpha=1, line_kws=kde_kws, label='High Income Records')
    
    h2 = sns.histplot(confidence_array[indices_by_y_values[0], 1][:], kde=True, color=color_code_2, ax=grid.ax_marg_x, bins=bins_x, fill=True, alpha=1, line_kws=kde_kws, label='Low Income Records')
    # sns.histplot(y=confidence_array[indices_by_y_values[0], 0][:], kde=False, color="red", ax=grid.ax_marg_y, bins=bins_y, fill=True, alpha=0., line_kws=kde_kws)


    sns.regplot(x=confidence_array[indices_by_y_values[1], 1][:], y=confidence_array[indices_by_y_values[1], 0][:], ax=grid.ax_joint, scatter=False, color="black", line_kws=dict(linewidth=2, alpha=1, linestyle='--'))
    sns.regplot(x=confidence_array[indices_by_y_values[0], 1][:], y=confidence_array[indices_by_y_values[0], 0][:], ax=grid.ax_joint, scatter=False, color="black", line_kws=dict(linewidth=2, linestyle='-.'))


    
    # cbar1 = plt.colorbar(hb1, ax=grid.ax_joint, orientation='vertical')
    # cbar1.set_label('Density for Positive Output')
    # cbar2 = plt.colorbar(hb2, ax=grid.ax_joint, orientation='horizontal')
    # cbar2.set_label('Density for Negative Output')

    # g = sns.JointGrid(x = confidence_array[:, 1], y = confidence_array[:, 0])
    # g.plot(plt.hexbin, sns.histplot)

    # g = sns.jointplot(x = confidence_array[:, 1], y = confidence_array[:, 0], hue=y)
        # sns.regplot(x = confidence_array[indices_by_y_values[y_value], 1][:num_points], y = confidence_array[indices_by_y_values[y_value], 0][:num_points], gridsize=50, cmap='Reds', mincnt=1, vmin=1, vmax=10, label=y_names[y_value])
        # plt.show()

    plt.xlim(0.475, 1.025)
    plt.ylim(0.475, 1.025)
    handles_1, labels_1 = h1.get_legend_handles_labels()
    handles_2, labels_2 = h2.get_legend_handles_labels()
    grid.ax_joint.legend(handles=handles_1+handles_2, labels=labels_1+labels_2, loc='upper center', bbox_to_anchor=(0.5, 0.2))
    plt.xlabel('Confidence Score when queried with Single')
    plt.ylabel('Confidence Score when queried with Married')
    # g.ax_joint.legend_.remove()
    # plt.colorbar(label='Density')
    # plt.legend()
    # plt.show()

def draw_hexbin(experiment, confidence_array, y):
    y_values = np.array(experiment.ds.ds.meta["y_values"]).astype(int)
    indices_by_y_values = {y_value: np.where(y.ravel() == y_value)[0] for y_value in y_values}
    colors = ['Reds', 'Blues']
    y_names = ['Records with negative output', 'Records with positive output']
    plt.figure(figsize=(6, 5))
    for y_value in [1, 0]:
        plt.hexbin(x = confidence_array[indices_by_y_values[y_value], 1][:], y = confidence_array[indices_by_y_values[y_value], 0][:], gridsize=50, cmap=colors[y_value], mincnt=1, vmin=1, vmax=10, label=y_names[y_value])
        # sns.regplot(x = confidence_array[indices_by_y_values[y_value], 1][:num_points], y = confidence_array[indices_by_y_values[y_value], 0][:num_points], gridsize=50, cmap='Reds', mincnt=1, vmin=1, vmax=10, label=y_names[y_value])
        # plt.show()

    # plt.ylim(0.5, 1)
    plt.xlabel('Confidence Score when queried with sensitive attribute is set to positive')
    plt.ylabel('Conf. Score when queried with sensitive attribute is set to negative')
    # plt.colorbar(label='Density')
    # plt.legend()
    # plt.show()

def get_slopes(experiment, confidence_array, y):
    y_values_labels = np.array(experiment.ds.ds.meta["y_values"])
    y_values = np.arange(len(y_values_labels))
    y_values_labels = dict(zip(y_values, y_values_labels))
    indices_by_y_values = {y_value: np.where(y.ravel() == y_value)[0] for y_value in y_values}
    # print(y.ravel())

    return [np.polyfit(confidence_array[indices_by_y_values[y_value], 1], confidence_array[indices_by_y_values[y_value], 0], 1)[0] for y_value in y_values]


def get_angular_difference(experiment, confidence_array, y):
    slopes = get_slopes(experiment, confidence_array, y)
    return np.arctan(np.abs(slopes[1] - slopes[0]) / (1 + slopes[1] * slopes[0])) * np.sign(slopes[1] - slopes[0])


def calculate_stds(experiment, confidence_array, y):
    y_values = np.array(experiment.ds.ds.meta["y_values"]).astype(int)
    indices_by_y_values = {y_value: np.where(y.ravel() == y_value)[0] for y_value in y_values}

    return np.array([np.std(confidence_array[indices_by_y_values[y_value]], axis=0) for y_value in y_values])


    