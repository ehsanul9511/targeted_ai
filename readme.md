Datasets:
census19.csv contains the Census-19 dataset.
Adult_35222.csv and Adult_10000.csv contains the Adult dataset partitions for training and testing respectively.
texas_100_cleaned.csv contains the Texas-100X dataset.

Codebase:
Source Files:
data_utils.py contains code to load dataset and preprocess data. Also contains code to sample data matching target correlation at a subgroup level as described in section 6.1 paragraph 'Sampling Technique'.
model_utils.py contains code to define the model architecture and training hyperparameters.
experiment_utils.py contains code to setup experiment for a particular scenario.
attack_utils.py contains implementation of existing attacks including CSMIA, LOMIA, imputation attack, and Neuron Importance Attack.
whitebox_attack.py contains helper functions needed to perform the neuron importance attack.
disparity_inference_utils.py contains implementation of Confidence Matrix generation (Algorithm 1), Angular Difference computation (Algorithm 2).
targeted_inference.py contains implementation of the targeted attribute inference attacks (section 5.3).
bcorr_utils.py contains implementation of the sampling stage of the BCorr defense (section 7.2).

Notebooks:
correlation_vs_attack_performance.ipynb contains the code to run the experiment described in section 4.1 which shows the strong connection between correlation and attack performance.
angular_difference_by_sex.ipynb contains the code to run the experiment described in section 4.2 which shows how the angular difference can be used to identify vulnerable groups.
imputation_vs_ai_aux_size_and_distrib_diff.ipynb contains the code to run the experiment described in section 6.2 which compares the performance between ideal imputation attack and practical imputation attack.
disparity_inference.ipynb contains the code to run the experiment described in section 6.3 which shows how the disparity inference attack can be used to rank groups based on their vulnerability.
targeted_attribute_inference.ipynb contains the code to run the experiments in section 6.4 which shows how the targeted attribute inference attack outperform their untargeted counterparts and practical imputation attacks.
bcorr_defense.ipynb contains the code to run the experiments in section 7.2 which shows how the BCorr defense can be used to mitigate the vulnerability of the model to the targeted attribute inference attack.
