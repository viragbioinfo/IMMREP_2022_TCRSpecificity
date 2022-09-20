"""
Train a k-nearest neighbours (KNN) classifier using TCRdist distances
over the ImmRep2022 training data and test over test data.
Training and testing are done separately for task1 and task2.
The task2 folder has results from 2 types of models.
    @ The files that end with "binary" contain predictions from each epitope's
    binary model (trained for task 1).
    @ The files that end with "multi" are created by getting predictions from a multiclass model
    trained on only epitope-specific TCRs from the training data files.

The script is assumed to be placed in the following directories tree:

python_projects/
    IMMREP_2022_TCRSpecificity/
        methods_scripts/
            immrep_tcrdist/
                immrep_tcrdist.py     <----------

Script results will be written to new folders created under the immrep_tcrdist folder (where the script is located).

tcrdist3 package must be installed in the environment. You can use manual installation or a docker container.
Full installation instructions are available at:
https://tcrdist3.readthedocs.io/en/latest/index.html

*If using the tcrdist3 docker container, the sklearn package should be pip installed in it as well.
**If using a different location for the script, just make sure to change paths
to the train and test data folders under the Params section:
paths.input_training_data_folder, paths.input_test_data_folder
(or change paths.folder)

Good luck!
Liel Cohen-Lavi

"""

######## --------------- Imports ------------- ######## <editor-fold>

import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
import math
from datetime import datetime
import sys
import scipy.stats as stats
import time

import matplotlib.patches as patches
from matplotlib.patches import Patch

from tcrdist.repertoire import TCRrep

from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve

import pickle

######## </editor-fold>

######## --------------- Functions ------------- ######## <editor-fold>

def create_folder(path):
    ''' If folder doesn't exist, create it '''
    if not os.path.exists(path):
        os.makedirs(path)

    return str(path)

def list_remove_instance(list1, itemToRemove):
    """ Remove itemToRemove from list1 if it exists.
        Removes all of its instances!
    """
    new_list = [item for item in list1 if item != itemToRemove]
    return new_list

def df_drop_duplicates(df, subset=None, keep='first'):
    """
    Get a pandas dataframe, drop duplicate rows, and return a copy of the reduced dataframe
    as well as the number of rows that were dropped.

    @param df: pandas dataframe
    @param subset: list of col names or None. Only consider certain columns
                   for identifying duplicates, by default (None) use all of the columns.
    @param keep: Determines which duplicates (if any) to keep.
                - first : Drop duplicates except for the first occurrence.
                - last : Drop duplicates except for the last occurrence.
                - False : Drop all duplicates.
                Default 'first'
    @return: pandas dataframe, int
    """
    rows_before = df.shape[0]
    df_after = df.drop_duplicates(subset=subset, keep=keep).copy()
    rows_after = df_after.shape[0]

    return df_after, rows_before-rows_after

def get_time_from_start(start_time, sec_or_minute='minute'):
    """
    Returns time that passed since start time
    in seconds or minutes.

    :param start_time: start time (from time.time() )
    :param sec_or_minute: 'second' or 'minute'
    :return: float - time in seconds or minutes
    """
    if sec_or_minute== 'minute':
        units = 60.0
    elif sec_or_minute== 'second':
        units = 1.0
    else: raise Exception('unknown second_or_minute value')

    return (time.time() - start_time) / units

def get_str_time_from_start(start_time, sec_or_minute='minute'):
    """
    Returns time that passed since start time, as a string,
    in seconds or minutes format.

    :param start_time: start time (from time.time() )
    :param sec_or_minute: 'second' or 'minute'
    :return: string - time in seconds or minutes
    """
    time_from = get_time_from_start(start_time, sec_or_minute=sec_or_minute)

    return ("%2.4f %ss" % (time_from, sec_or_minute))

class Object(object):
    pass

def write_string_to_txt_file(string, file_path):
    text_file = open(file_path, "w")
    text_file.write(string)
    text_file.close()

def get_train_data(input_data_folder, drop_duplicates_within_ep=False,
                   duplicates_within_ep_cols=['TRB_CDR3', 'TRA_CDR3', 'TRAV', 'TRAJ', 'TRBV', 'TRBJ', 'epitope'],
                   drop_duplicates_between_all=False,
                   duplicates_between_all_cols=['TRB_CDR3', 'TRA_CDR3', 'TRAV', 'TRAJ', 'TRBV', 'TRBJ'],
                   save_to_folder=None,
                   rename_cols=None,
                   add_count_col=False):
    """
    Gets path string input_data_folder and loads each epitope's data into a separate dataframe.
    Also, combines all dataframes into a single unifying dataframe,
    with added column 'epitope' (epitope str / negatives+epitope str),
    column 'epitope2' (epitope str / negatives='none' str),
    column 'subject' all with 'unknown'.
    If boolean drop_duplicates_within_ep=True, drops duplicates within each epitope's df,
    based on columns in duplicates_within_ep_cols (a list of column names).
    Based on columns in duplicates_between_all_cols (a list of column names),
    also checks for duplicates between all epitopes (in the unifying df only).
    However, drops them from df only if boolean drop_duplicates_between_all=True.
    *Please make sure that duplicates_between_all_cols does not contain an "epitope" column.

    @return: a dictionary with each epitope's df, the unified df, a list of epitopes
    """
    data_files = os.listdir(input_data_folder)

    ep_dfs = {}
    num_rows_total = 0
    for file in data_files:
        if file != 'README.txt':
            ep = file.split('.')[0]
            ep_df = pd.read_csv(os.path.join(input_data_folder, file), sep='\t')

            ep_df['epitope'] = None
            ep_df.loc[ep_df['Label'] == 1, 'epitope'] = ep
            ep_df.loc[ep_df['Label'] == -1, 'epitope'] = 'none'

            ep_df['epitope2'] = None
            ep_df.loc[ep_df['Label'] == 1, 'epitope2'] = ep
            ep_df.loc[ep_df['Label'] == -1, 'epitope2'] = 'Neg - ' + ep

            ep_df['subject'] = 'none'

            if add_count_col:
                ep_df['count'] = 1

            if drop_duplicates_within_ep:
                rows_before = ep_df.shape[0]

                ### Drop duplicates by only positive examples, then by only negative examples, then
                ### by both together (to see if there are identical TCRs between positive and negative)
                ### in order to get the counts
                ep_df_pos = ep_df.loc[ep_df['Label'] == 1].copy()
                ep_df_neg = ep_df.loc[ep_df['Label'] == -1].copy()
                assert ep_df_pos.shape[0] + ep_df_neg.shape[0] == ep_df.shape[0]

                ep_df_pos_new, num_pos_dropped = df_drop_duplicates(ep_df_pos, subset=duplicates_within_ep_cols, keep='first')
                ep_df_neg_new, num_neg_dropped = df_drop_duplicates(ep_df_neg, subset=duplicates_within_ep_cols, keep='first')
                ep_df_new = pd.concat([ep_df_pos_new, ep_df_neg_new])

                ep_df_new, num_combined_dropped = df_drop_duplicates(ep_df_new, subset=duplicates_within_ep_cols, keep='first')

                ### Drop duplicates for entire dataframe
                ep_df, num_dropped = df_drop_duplicates(ep_df, subset=duplicates_within_ep_cols, keep='first')
                assert ep_df_new.shape[0] == ep_df.shape[0]

                print(f'Epitope {ep}: Dropped {num_dropped} duplicates out of {rows_before}. Positives {num_pos_dropped}, negatives {num_neg_dropped}, cross {num_combined_dropped}. by cols {duplicates_within_ep_cols}')

            if rename_cols is not None:
                ep_df = ep_df.rename(columns=rename_cols)

            ep_dfs[ep] = ep_df
            num_rows_total += ep_df.shape[0]

            if save_to_folder is not None:
                ep_df.to_csv(os.path.join(save_to_folder, f'epitope_{ep}.csv'))

    df_all = pd.concat(list(ep_dfs.values()))
    assert df_all.shape[0] == num_rows_total

    ### Check for duplicates between all epitopes together
    df_all_pos = df_all.loc[df_all['Label'] == 1].copy()
    df_all_neg = df_all.loc[df_all['Label'] == -1].copy()

    if rename_cols is not None:
        updated_list = []
        for col in duplicates_between_all_cols:
            if col in rename_cols:
                updated_list.append(rename_cols[col])
            else:
                updated_list.append(col)

        duplicates_between_all_cols = updated_list

    duplicates_between_all_cols = list_remove_instance(duplicates_between_all_cols, 'epitope')
    duplicates_between_all_cols = list_remove_instance(duplicates_between_all_cols, 'Epitope')

    df_all_pos_new, num_pos_dropped = df_drop_duplicates(df_all_pos,
                                                   subset=duplicates_between_all_cols, keep='first')
    df_all_neg_new, num_neg_dropped = df_drop_duplicates(df_all_neg,
                                                   subset=duplicates_between_all_cols, keep='first')
    df_all_new = pd.concat([df_all_pos_new, df_all_neg_new])
    df_all_new, num_all_cross_dropped = df_drop_duplicates(df_all_new,
                                                   subset=duplicates_between_all_cols, keep='first')

    print(f'\nNum duplicates between epitopes: {num_pos_dropped} of {df_all_pos.shape[0]}, between negatives: {num_neg_dropped} of {df_all_neg.shape[0]}. '
          f'\nCross duplicates between pos & neg (after dropping pos & neg separately): {num_all_cross_dropped}')

    ### Drop them if asked.
    if drop_duplicates_between_all:
        df_all = df_all_new
        print('Dropped duplicates from df_all!')
    else:
        print('*Did not drop duplicates from df_all.')

    return ep_dfs, df_all, list(ep_dfs.keys())

def find_best_roc_auc_cutoff(y_true, y_predicted):
    """ Find the best probability cutoff point for
    predictions of a binary classification model,
    by the Youden's J index:
            max(t) {Sensitivity(t) + Specificity(t) − 1} =
            max(t) {true positive rate (t) - false positive rate (t)}
    parameters: y_true - vector of true binary labels (0/1)
                y_predicted - vector of predicted probabilities (0-1)
    returns: prediction threshold
            (positive labels should be then predicted
            if probability >= threshold)
    """
    fpr, tpr, threshold = roc_curve(y_true, y_predicted)
    calc = pd.DataFrame({'tpr-fpr': tpr - fpr, 'threshold': threshold})
    ind_measure_max = calc['tpr-fpr'].idxmax()
    best_threshold = calc.loc[ind_measure_max, 'threshold']

    return best_threshold

def get_class_preds_sklearn_model(model, preds_proba, class_name, raise_exception=True):
    """
    Gets a sklearn model, a matrix of predicted probabilities (output from function model.predict_proba)
    and the class identifier (string/int. according to the model y_train vector).
    Returns the vector of predictions for the specified class, from preds_proba matrix.
    If the class is not in the model classes list, will return none or raise an error.
    @param model: sklearn model object
    @param preds_proba: predicted class probabilities, output from function model.predict_proba
    @param class_name: target class identifier (string/int. according to the model y_train vector)
    @param raise_exception: boolean. Whether to raise an error if the requested class is
                            not in the model classes list.
    @return: the vector of predictions for the specified class, from preds_proba
    """
    model_classes = model.classes_

    class_ind = None
    for i, cl in enumerate(model_classes):
        if cl==class_name:
            class_ind = i

    if class_ind is not None:
        return preds_proba[:, class_ind]
    else:
        if raise_exception:
            raise Exception('class_name was not found in the given model classes.')
        else:
            return None

######## </editor-fold>

######## --------------- Params ------------- ######## <editor-fold>

# paths
paths = Object()

paths.folder = os.getcwd()
paths.main_immrep_folder = os.path.dirname(os.path.dirname(paths.folder))

paths.input_training_data_folder = os.path.join(paths.main_immrep_folder, 'training_data') # this folder should contain the files from https://github.com/viragbioinfo/IMMREP_2022_TCRSpecificity/tree/main/training_data
paths.input_test_data_folder = os.path.join(paths.main_immrep_folder, 'test_set') # this folder should contain the files from https://github.com/viragbioinfo/IMMREP_2022_TCRSpecificity/tree/main/test_set

paths.out_ep_spec_proc_train_folder = os.path.join(paths.folder, 'ep_spec_processed_training_data/') # for writing the training data files in a tcrdist format
paths.out_ep_spec_models_folder = os.path.join(paths.folder, 'ep_spec_trained_models/')

paths.out_proc_test_data_folder = os.path.join(paths.folder, 'processed_test_data_task1/') # for writing the test data files in a tcrdist format
paths.out_test_task1_folder = os.path.join(paths.folder, 'test_task1/')

paths.out_train_folder_task2 = os.path.join(paths.folder, 'processed_test_data_task2/') # for writing the training data file for task2 in a tcrdist format
paths.out_test_task2_folder = os.path.join(paths.folder, 'test_task2/') # for output of the task2 test results

create_folder(paths.out_ep_spec_proc_train_folder)
create_folder(paths.out_ep_spec_models_folder)
create_folder(paths.out_test_task1_folder)
create_folder(paths.out_proc_test_data_folder)
create_folder(paths.out_train_folder_task2)
create_folder(paths.out_test_task2_folder)

# script params
k_nbrs_bin_models = 5 # number of neighbours for epitope specific models (task1 and task2-binary)
k_nbrs_task2_multi = 3 # number of neighbours for multiclass model (task 2)
retrain_tcrdist_bin_models = True # retrain models for test_task1 and test_task2_binary
retrain_tcrdist_multi_model = True # retrain model for test_task2_multi
perform_cv_for_estimation = True # estimate performance of task1 binary models & task2 multiclass model by cross validation on train set
cv_folds = 5 # cross validation number of folds
test_task1 = True
test_task2_binary = True
test_task2_multi = True

######## </editor-fold>

######## --------------- Get training data for tcrdist ------------- ######## <editor-fold>

rename_cols = {'TRAV': 'v_a_gene', 'TRAJ': 'j_a_gene',
               'TRBV': 'v_b_gene', 'TRBJ': 'j_b_gene',
               'TRA_CDR3': 'cdr3_a_aa', 'TRB_CDR3': 'cdr3_b_aa'}
# Get the training data for each epitope from the separate files, write to new files after processing
# and also get the unified dataframe that contains data for all epitopes combined.
ep_dfs, df_all, epitopes = get_train_data(paths.input_training_data_folder, drop_duplicates_within_ep=True,
                                          duplicates_within_ep_cols=['TRB_CDR3', 'TRA_CDR3', 'TRAV', 'TRAJ', 'TRBV', 'TRBJ', 'epitope'],
                                          drop_duplicates_between_all=False,
                                          duplicates_between_all_cols=['TRB_CDR3', 'TRA_CDR3', 'TRAV', 'TRAJ', 'TRBV', 'TRBJ'],
                                          save_to_folder=paths.out_ep_spec_proc_train_folder,
                                          rename_cols=rename_cols,
                                          add_count_col=True)

######## </editor-fold>

tcrreps = {}
models = {}
dist_mats = {}

for epitope in epitopes:
    print(f'\n###### --------------- Epitope {epitope} --------------- ######')

    ######## --------------- get binary (epitope-specific) models ------------- ######## <editor-fold>

    ### paths
    ep_paths = Object()
    ep_paths.ep_train_data_filename = os.path.join(paths.out_ep_spec_proc_train_folder, f'epitope_{epitope}.csv')

    # epitope's output folder and file names
    ep_paths.ep_model_folder = os.path.join(paths.out_ep_spec_models_folder, epitope)
    ep_paths.ep_tcrrep_filename = os.path.join(ep_paths.ep_model_folder, epitope + '_tcrrep')
    ep_paths.ep_model_filename = os.path.join(ep_paths.ep_model_folder, epitope + '_model')
    ep_paths.ep_dist_filename = os.path.join(ep_paths.ep_model_folder, epitope + '_distances')
    ep_paths.ep_cv5_res_filename = os.path.join(ep_paths.ep_model_folder, epitope + '_train_cv5_performance.txt')
    create_folder(ep_paths.ep_model_folder)

    # train / get model
    if retrain_tcrdist_bin_models or not os.path.exists(ep_paths.ep_model_filename): # if asked to retrain model or model file does not exist
        train_start_time = time.time()

        train_df = ep_dfs[epitope]
        tcrrep = TCRrep(cell_df=train_df.reset_index(drop=True),
                        organism='human', chains=['alpha','beta'],
                        compute_distances=True,
                        deduplicate=False)
        X = pd.DataFrame(tcrrep.pw_alpha + tcrrep.pw_beta) # training data distances matrix
        Y = pd.DataFrame(tcrrep.cell_df['epitope']) # training data labels

        assert X.shape[0] == X.shape[1] # X is nXn
        assert X.shape[0] == Y.shape[0] # Y length = n
        for i in range(X.shape[0]):
            assert X.index[i] == Y.index[i], f'index number {i} unequal: X {X.index[i]}, Y {Y.index[i]}' # X, Y indices are equal

        ######## --------------- Assess performance on training set and tune threshold by cross validation ------------- ######## <editor-fold>

        if perform_cv_for_estimation:
            # Get 5-CV folds
            kfolds = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            kfolds.get_n_splits(X)

            # metrics storing dicts
            recall_cv, precision_cv, accuracy_cv, threshold_cv = [], [], [], []

            for train_ind, test_ind in kfolds.split(X, Y):

                X_train, X_test = X.iloc[train_ind, train_ind], X.iloc[test_ind, train_ind]
                Y_train, Y_test = Y.iloc[train_ind], Y.iloc[test_ind]

                model_fold = KNeighborsClassifier(n_neighbors=k_nbrs_bin_models, metric='precomputed', weights='distance')
                model_fold.fit(X_train, Y_train.values.ravel())
                predictions = model_fold.predict(X_test)

                accuracy_cv.append(accuracy_score(Y_test, predictions))
                precision_cv.append(precision_score(Y_test, predictions, average='macro')) # macro - simplest, unweighted mean between all classes
                recall_cv.append(recall_score(Y_test, predictions, average='macro')) # macro - simplest, unweighted mean between all classes

                pred_prob = get_class_preds_sklearn_model(model_fold, model_fold.predict_proba(X_test), epitope)
                threshold_cv.append(find_best_roc_auc_cutoff(Y_test.replace({epitope: 1, 'none': 0}), pred_prob))

            cv_res = f'\n{epitope}: average CV5 accuracy: {np.mean(accuracy_cv):.3f}, recall: {np.mean(recall_cv):.3f}, precision: {np.mean(precision_cv):.2f}'
            print(cv_res)
            print(f'thresholds: {threshold_cv}. Average: {np.mean(threshold_cv):.3f}')
            write_string_to_txt_file(cv_res, ep_paths.ep_cv5_res_filename)

        ######## </editor-fold>

        # train model (on entire train set)
        model = KNeighborsClassifier(n_neighbors=k_nbrs_bin_models, metric='precomputed', weights='distance')
        model.fit(X, Y.values.ravel())

        # save tcrrep object to file
        with open(ep_paths.ep_tcrrep_filename, 'wb') as f:
            pickle.dump(tcrrep, f)

        # save model to file
        with open(ep_paths.ep_model_filename, 'wb') as f:
            pickle.dump(model, f)

        # save distance matrix to file
        with open(ep_paths.ep_dist_filename, 'wb') as f:
            pickle.dump(X, f)

        print(f'\n{epitope}: Trained model and saved to file.\n')

    else:
        # get tcrrep object from file
        with open(ep_paths.ep_tcrrep_filename, 'rb') as f:
            tcrrep = pickle.load(f)

        # get model from file
        with open(ep_paths.ep_model_filename, 'rb') as f:
            model = pickle.load(f)

        # get best distance matrix from file
        with open(ep_paths.ep_dist_filename, 'rb') as f:
            X = pickle.load(f)

        print('# Loaded model from file.')

    # save model and distance matrix to dictionaries, for task 2 testing
    tcrreps[epitope] = tcrrep
    models[epitope] = model
    dist_mats[epitope] = X

    ######## </editor-fold>

    ######## --------------- test - task 1 - binary models ------------- ######## <editor-fold>

    if test_task1:
        # paths
        ep_paths.input_test_ep_filename_orig = os.path.join(paths.input_test_data_folder, epitope + '.txt')
        ep_paths.input_test_ep_filename_csv = os.path.join(paths.out_proc_test_data_folder, epitope + '.csv')
        ep_paths.output_test_ep_filename = os.path.join(paths.out_test_task1_folder, epitope + '.txt')

        print('\n### Testing model! (task1)')
        # test data import
        ep_test_df_orig = pd.read_csv(ep_paths.input_test_ep_filename_orig, sep='\t')
        print(f'Test data size: {ep_test_df_orig.shape[0]} samples')

        ep_test_df = ep_test_df_orig.copy()
        ep_test_df['count'] = 1
        ep_test_df = ep_test_df.rename(columns=rename_cols)

        # write orig test file to csv (tcrdist format)
        ep_test_df.to_csv(ep_paths.input_test_ep_filename_csv)

        # get prediction for test data
        tcrrep_test = TCRrep(cell_df=ep_test_df,
                    organism='human', chains=['alpha','beta'],
                    compute_distances=False,
                    deduplicate=False)
        # make sure no TCRs were dropped
        assert (tcrrep_test.cell_df.index == ep_test_df.index).sum() == ep_test_df.shape[0]

        # get distances between test and train
        tcrrep_test.compute_rect_distances(df=tcrrep_test.clone_df, df2=tcrrep.clone_df) # for large data (>10k) can be replaced with compute_sparse_rect_distances
        X_test = pd.DataFrame(tcrrep_test.rw_alpha + tcrrep_test.rw_beta) # distances matrix

        # assert dimensions are n_test X n_train
        assert X_test.shape[0] == ep_test_df.shape[0]
        assert X_test.shape[1] == X.shape[0]

        # perform test
        preds = get_class_preds_sklearn_model(model, model.predict_proba(X_test), epitope)
        print(f'\nGot {ep_test_df.shape[0] - np.isnan(preds).sum()} predictions, out of requested {ep_test_df.shape[0]}')

        # Write test results to file in original format (.txt)
        ep_test_df_orig['preds'] = preds
        ep_test_df_orig.to_csv(ep_paths.output_test_ep_filename, sep='\t')

    ######## </editor-fold>

######## --------------- tcrdist test - task 2 - get data ------------- ######## <editor-fold>

if test_task2_binary or test_task2_multi:
    # paths
    paths.task2_test_data_filename = os.path.join(paths.input_test_data_folder, 'testSet_Global_NoHeaders.txt')
    paths.task2_test_data_filename_csv = os.path.join(paths.out_test_task2_folder, 'testSet_Global_NoHeaders.csv')

    # read task2 test file
    df_test_task2_orig = pd.read_csv(paths.task2_test_data_filename, sep='\t')

    # tweak to fit tcrdist input df structure and save as csv
    df_test_task2 = df_test_task2_orig.copy()
    df_test_task2['count'] = 1
    df_test_task2 = df_test_task2.rename(columns=rename_cols)
    df_test_task2.to_csv(paths.task2_test_data_filename_csv)

######## </editor-fold>

######## --------------- tcrdist test - task 2 (combine binary models for multiclass prediction) ------------- ######## <editor-fold>

if test_task2_binary:
    print('\n### Performing task2 testing based on binary models')
    # paths
    paths.task2_test_res_bin_file = os.path.join(paths.out_test_task2_folder,
                                                 'testSet_Global_NoHeaders_results_fr_binary.txt')
    paths.task2_test_res_bin_longform_file = os.path.join(paths.out_test_task2_folder,
                                                          'testSet_Global_NoHeaders_results_longform_fr_binary.csv')

    # get predictions from each epitopes model and add to results df
    df_test_task2_bin_res = df_test_task2_orig.copy() # for storing results

    for epitope in epitopes:
        model = models[epitope]
        tcrrep_train = tcrreps[epitope]
        X_train = dist_mats[epitope]

        # get tcrrep object for test data
        tcrrep_test = TCRrep(cell_df=df_test_task2,
                             organism='human', chains=['alpha','beta'],
                             compute_distances=False,
                             deduplicate=False)
        # make sure no TCRs were dropped
        assert (tcrrep_test.cell_df.index == df_test_task2.index).sum() == df_test_task2.shape[0]

        # get distances between test and train
        tcrrep_test.compute_rect_distances(df=tcrrep_test.clone_df, df2=tcrrep_train.clone_df) # for large data (>10k) can be replaced with compute_sparse_rect_distances
        X_test = pd.DataFrame(tcrrep_test.rw_alpha + tcrrep_test.rw_beta) # distances matrix

        # assert dimensions are n_test X n_train
        assert X_test.shape[0] == df_test_task2.shape[0]
        assert X_test.shape[1] == tcrrep_train.clone_df.shape[0]

        # perform test and add to df
        preds = get_class_preds_sklearn_model(model, model.predict_proba(X_test), epitope)
        print(f'{epitope}: got {df_test_task2.shape[0] - np.isnan(preds).sum()} predictions, out of requested {df_test_task2.shape[0]}')
        df_test_task2_bin_res[epitope] = preds

    # Write test results to file in original format (.txt)
    df_test_task2_bin_res['predicted_epitope'] = df_test_task2_bin_res[epitopes].idxmax(axis=1)
    df_test_task2_bin_res.to_csv(paths.task2_test_res_bin_file, sep='\t')

    #### Reformat the task2 test results to a long form table with all epitopes scores, and save to file
    df_test_task2_bin_res['TCR_id'] = df_test_task2_bin_res.index
    df_test_task2_bin_res = df_test_task2_bin_res.drop(columns='predicted_epitope')

    task2_test_bin_long = df_test_task2_bin_res.melt(id_vars=['TCR_id']+list(df_test_task2_orig.columns),
                                                     value_name='score', var_name='predicted')
    task2_test_bin_long = task2_test_bin_long.sort_values('TCR_id')
    task2_test_bin_long['rank'] = np.nan
    task2_test_bin_long.reset_index(inplace=True, drop=True)

    for tcr_id in range(df_test_task2_orig.shape[0]):
        # rank epitope scores for a specific TCR
        tcr_df = task2_test_bin_long.loc[task2_test_bin_long['TCR_id'] == tcr_id].copy()
        tcr_df['rank'] = tcr_df['score'].rank(method='min', ascending=False)
        # get these ranks to the main df
        task2_test_bin_long.loc[task2_test_bin_long['TCR_id'] == tcr_id, 'rank'] = tcr_df['rank']

    assert task2_test_bin_long['rank'].isna().sum() == df_test_task2_bin_res[epitopes].isna().sum().sum()

    task2_test_bin_long.to_csv(paths.task2_test_res_bin_longform_file, index=False)

######## </editor-fold>

######## --------------- tcrdist task 2 - multiclass model ------------- ######## <editor-fold>

if test_task2_multi:
    # paths
    paths.task2_multi_train_file = os.path.join(paths.out_train_folder_task2, 'task2_multi_training_data.csv') # for saving training df
    paths.task2_multi_model_file = os.path.join(paths.out_train_folder_task2, 'task2_multi_model')
    paths.task2_multi_tcrrep_file = os.path.join(paths.out_train_folder_task2, 'task2_multi_tcrrep')
    paths.task2_multi_distmat_file = os.path.join(paths.out_train_folder_task2, 'task2_multi_distmat')
    paths.task2_multi_epitope_map_file = os.path.join(paths.out_train_folder_task2, 'task2_multi_epitope_mapping')
    paths.task2_multi_cv_res = os.path.join(paths.out_train_folder_task2, 'task2_multi_cv_res')

    paths.task2_test_res_multi_file = os.path.join(paths.out_test_task2_folder,
                                                   'testSet_Global_NoHeaders_results_fr_multi.txt')
    paths.task2_test_res_multi_longform_file = os.path.join(paths.out_test_task2_folder,
                                                            'testSet_Global_NoHeaders_results_longform_fr_multi.csv')

    ######## --------------- tcrdist Task 2 - multiclass - training ------------- ######## <editor-fold>

    print('\n### Getting task2 multiclass model')

    if retrain_tcrdist_multi_model or not os.path.exists(paths.task2_multi_model_file): # if asked to retrain model or model file does not exist

        # Get a unified training set - only positive examples of all epitopes
        df_train_task2_multi = df_all[df_all['epitope'] != 'none']
        df_train_task2_multi.to_csv(paths.task2_multi_train_file)
        df_train_task2_multi.epitope.value_counts().to_csv(os.path.join(paths.out_train_folder_task2, 'task2_multi_train_data_summary.csv'))

        # get tcrrep object, X (distances) and Y (epitopes)
        tcrrep_t2mult = TCRrep(cell_df=df_train_task2_multi.reset_index(drop=True),
                        organism='human', chains=['alpha','beta'],
                        compute_distances=True,
                        deduplicate=False)
        X_t2mult = pd.DataFrame(tcrrep_t2mult.pw_alpha + tcrrep_t2mult.pw_beta) # training data distances matrix
        Y_t2mult = pd.DataFrame(tcrrep_t2mult.cell_df['epitope'].reset_index(drop=True)) # training data labels

        assert X_t2mult.shape[0] == X_t2mult.shape[1] # X is nXn
        assert X_t2mult.shape[0] == Y_t2mult.shape[0] # len(Y) = n
        for i in range(X_t2mult.shape[0]):
            assert X_t2mult.index[i] == Y_t2mult.index[i], f'index number {i} unequal: X {X_t2mult.index[i]}, Y {Y_t2mult.index[i]}' # X, Y indices are equal

        ######## --------------- Assess performance on training set by cross validation ------------- ######## <editor-fold>

        if perform_cv_for_estimation:
            # Get 5-CV folds
            kfolds = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            kfolds.get_n_splits(X_t2mult)

            # metrics storing dicts
            recall_cv, precision_cv, accuracy_cv = [], [], []

            for train_ind_fold, test_ind_fold in kfolds.split(X_t2mult, Y_t2mult):

                X_train_fold, X_test_fold = X_t2mult.iloc[train_ind_fold, train_ind_fold], X_t2mult.iloc[test_ind_fold, train_ind_fold]
                Y_train_fold, Y_test_fold = Y_t2mult.iloc[train_ind_fold], Y_t2mult.iloc[test_ind_fold]

                model_fold = KNeighborsClassifier(n_neighbors=k_nbrs_task2_multi, metric='precomputed', weights='distance')
                model_fold.fit(X_train_fold, Y_train_fold.values.ravel())
                predictions = model_fold.predict(X_test_fold)

                accuracy_cv.append(accuracy_score(Y_test_fold, predictions))
                precision_cv.append(precision_score(Y_test_fold, predictions, average='macro')) # macro - simplest, unweighted mean between all classes
                recall_cv.append(recall_score(Y_test_fold, predictions, average='macro')) # macro - simplest, unweighted mean between all classes

            cv_res = f'\nTask2 multi: average CV5 accuracy: {np.mean(accuracy_cv):.3f}, recall: {np.mean(recall_cv):.3f}, precision: {np.mean(precision_cv):.2f}'
            print(cv_res)
            write_string_to_txt_file(cv_res, paths.task2_multi_cv_res)

        ######## </editor-fold>

        # train model (on entire train set)
        model_t2mult = KNeighborsClassifier(n_neighbors=k_nbrs_task2_multi, metric='precomputed', weights='distance')
        model_t2mult.fit(X_t2mult, Y_t2mult.values.ravel())

        # save tcrrep object to file
        with open(paths.task2_multi_tcrrep_file, 'wb') as f:
            pickle.dump(tcrrep_t2mult, f)

        # save model to file
        with open(paths.task2_multi_model_file, 'wb') as f:
            pickle.dump(model_t2mult, f)

        # save distance matrix to file
        with open(paths.task2_multi_distmat_file, 'wb') as f:
            pickle.dump(X_t2mult, f)

        print(f'\nTask2 multiclass: Trained model and saved to file.\n')

    else:
        # get tcrrep object from file
        with open(paths.task2_multi_tcrrep_file, 'rb') as f:
            tcrrep_t2mult = pickle.load(f)

        # get model from file
        with open(paths.task2_multi_model_file, 'rb') as f:
            model_t2mult = pickle.load(f)

        # get best distance matrix from file
        with open(paths.task2_multi_distmat_file, 'rb') as f:
            X_t2mult = pickle.load(f)

        print('# Loaded model from file.')

    ######## </editor-fold>

    ######## --------------- tcrdist Task 2 - multiclass - testing ------------- ######## <editor-fold>

    print('\n### Testing task2 multiclass model')

    df_test_task2_mult_res = df_test_task2_orig.copy() # for storing results

    # get tcrrep object for test data
    tcrrep_test = TCRrep(cell_df=df_test_task2,
                         organism='human', chains=['alpha','beta'],
                         compute_distances=False,
                         deduplicate=False)
    # make sure no TCRs were dropped
    assert (tcrrep_test.cell_df.index == df_test_task2.index).sum() == df_test_task2.shape[0]

    # get distances between test and train
    tcrrep_test.compute_rect_distances(df=tcrrep_test.clone_df, df2=tcrrep_t2mult.clone_df) # for large data (>10k) can be replaced with compute_sparse_rect_distances
    X_test = pd.DataFrame(tcrrep_test.rw_alpha + tcrrep_test.rw_beta) # distances matrix

    # assert dimensions are n_test X n_train
    assert X_test.shape[0] == df_test_task2.shape[0]
    assert X_test.shape[1] == tcrrep_t2mult.clone_df.shape[0]

    # perform test and add to df
    preds = model_t2mult.predict(X_test)
    print(f'Task2 multiclass: got {df_test_task2.shape[0] - pd.isna(pd.Series(preds)).sum()} predictions, out of requested {df_test_task2.shape[0]}')
    df_test_task2_mult_res['predicted_epitope'] = preds

    # Write test results to file in original format (.txt)
    df_test_task2_mult_res.to_csv(paths.task2_test_res_multi_file, sep='\t')

    #### Reformat the task2 test results to a long form table with all epitopes scores, and save to file
    preds_mat = model_t2mult.predict_proba(X_test)
    preds_mat = pd.DataFrame(preds_mat, columns=model_t2mult.classes_)

    df_test_task2_mult_res = df_test_task2_mult_res.join(preds_mat)
    df_test_task2_mult_res = df_test_task2_mult_res.drop(columns='predicted_epitope')
    df_test_task2_mult_res['TCR_id'] = df_test_task2_mult_res.index

    task2_test_mult_long = df_test_task2_mult_res.melt(id_vars=['TCR_id']+list(df_test_task2_orig.columns),
                                                     value_name='score', var_name='predicted')
    task2_test_mult_long = task2_test_mult_long.sort_values('TCR_id')
    task2_test_mult_long['rank'] = np.nan
    task2_test_mult_long.reset_index(inplace=True, drop=True)

    for tcr_id in range(df_test_task2_orig.shape[0]):
        # rank epitope scores for a specific TCR
        tcr_df = task2_test_mult_long.loc[task2_test_mult_long['TCR_id'] == tcr_id].copy()
        tcr_df['rank'] = tcr_df['score'].rank(method='min', ascending=False)
        # get these ranks to the main df
        task2_test_mult_long.loc[task2_test_mult_long['TCR_id'] == tcr_id, 'rank'] = tcr_df['rank']

    assert task2_test_mult_long['rank'].isna().sum() == df_test_task2_mult_res[epitopes].isna().sum().sum()

    task2_test_mult_long.to_csv(paths.task2_test_res_multi_longform_file, index=False)

######## </editor-fold>






