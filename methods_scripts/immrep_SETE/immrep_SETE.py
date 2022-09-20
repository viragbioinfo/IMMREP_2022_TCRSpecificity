"""
Train SETE over the ImmRep2022 training data and test over test data.
Training and testing are done separately for task1 and task2. For task1, a binary
classifier is trained for each epitope separately. For task2, a multiclass classifier
is trained over all epitope-specific TCRs in the training data.
The SETE method only uses the TCR CDR3-beta. Duplicates are removed from each epitope's
training set (based on CDR3-beta only).

The script is assumed to be placed in the following directories tree:

python_projects/
    IMMREP_2022_TCRSpecificity/
        methods_scripts/
            immrep_SETE/
                immrep_SETE.py     <----------
    SETE/
        SETE.py
        ...
    ...

Script results will be written to new folders created under the immrep_SETE folder (where the script is located).

Clone the SETE code from
"https://github.com/wonanut/SETE"
or, if my git pull request is not accepted yet, please clone repository
"https://github.com/liel-cohen/SETE" instead. (the original wonanut code will not work with this script!)

Please notice that the cloned SETE folder should be placed in the root folder
(parent folder of the IMMREP_2022_TCRSpecificity folder).
If the SETE folder is located elsewhere, please change the sete_package_folder variable in the
"Imports" part of the script, to the correct SETE folder string.

Good luck!
Liel Cohen-Lavi

"""

######## --------------- Imports ------------- ######## <editor-fold>

import os
import socket

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import random
import math
from datetime import datetime
import sys
import scipy.stats as stats
import time

import matplotlib.patches as patches
from matplotlib.patches import Patch

import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA

# add SETE project directory to paths to it will be importable
sete_package_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))), 'SETE') # if SETE folder is in the parent folder. If not, set the path manually
sys.path.append(sete_package_folder)
import SETE

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

def get_train_data(input_data_folder, drop_duplicates_within_ep=False,
                   duplicates_within_ep_cols=['TRB_CDR3', 'TRA_CDR3', 'TRAV', 'TRAJ', 'TRBV', 'TRBJ', 'epitope'],
                   drop_duplicates_between_all=False,
                   duplicates_between_all_cols=['TRB_CDR3', 'TRA_CDR3', 'TRAV', 'TRAJ', 'TRBV', 'TRBJ'],
                   save_to_folder=None,
                   rename_cols=None):
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

            # ep_df['subject'] = random.choices(['mock_sub_1', 'mock_sub_2', 'mock_sub_3', 'mock_sub_4', 'mock_sub_5'],
            #                                   k=ep_df.shape[0])
            ep_df['subject'] = 'none'

            if drop_duplicates_within_ep:
                rows_before = ep_df.shape[0]

                ### Drop duplicates by only positive examples, then by only negative examples, then
                ### by both together (to see if there are identical TCRs between positive and negative)
                ### in order to get the counts
                ep_df_pos = ep_df.loc[ep_df['Label'] == 1].copy()
                ep_df_neg = ep_df.loc[ep_df['Label'] == -1].copy()
                assert ep_df_pos.shape[0] + ep_df_neg.shape[0] == ep_df.shape[0]

                ep_df_pos_new, num_pos_dropped = df_drop_duplicates(ep_df_pos,
                                                                              subset=duplicates_within_ep_cols, keep='first')
                ep_df_neg_new, num_neg_dropped = df_drop_duplicates(ep_df_neg,
                                                                              subset=duplicates_within_ep_cols, keep='first')
                ep_df_new = pd.concat([ep_df_pos_new, ep_df_neg_new])

                ep_df_new, num_combined_dropped = df_drop_duplicates(ep_df_new,
                                                                               subset=duplicates_within_ep_cols, keep='first')

                ### Drop duplicates for entire dataframe
                ep_df, num_dropped = df_drop_duplicates(ep_df,
                                                                  subset=duplicates_within_ep_cols, keep='first')
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

class Object(object):
    pass

######## </editor-fold>

######## --------------- Params ------------- ######## <editor-fold>

# paths
paths = Object()

paths.SETE_folder = os.getcwd()
paths.main_immrep_folder = os.path.dirname(os.path.dirname(paths.SETE_folder))

# Data folders (in IMMREP_2022_TCRSpecificity folder)
paths.input_training_data_folder = os.path.join(paths.main_immrep_folder, 'training_data') # this folder should contain the files from https://github.com/viragbioinfo/IMMREP_2022_TCRSpecificity/tree/main/training_data
paths.input_test_data_folder = os.path.join(paths.main_immrep_folder, 'test_set') # this folder should contain the files from https://github.com/viragbioinfo/IMMREP_2022_TCRSpecificity/tree/main/test_set

# Output folders - task1
paths.task1_proc_train_data_folder = os.path.join(paths.SETE_folder, 'SETE_task1_training_data_processed/') # for writing the task1 precessed training data files in a SETE format
paths.task1_models_folder = os.path.join(paths.SETE_folder, 'SETE_task1_trained_models/')
paths.task1_proc_test_data_folder = os.path.join(paths.SETE_folder, 'SETE_task1_test_data_processed/') # for writing the test data files in a SETE format
paths.task1_test_res = os.path.join(paths.SETE_folder, 'SETE_task1_test_res/') # task 1 test results (binary models)

# Output folders - task2
paths.task2_train_folder = os.path.join(paths.SETE_folder, 'SETE_task2_train/') # for writing the training data file in a SETE format, and models
paths.task2_test_folder = os.path.join(paths.SETE_folder, 'SETE_task2_test/')

# create folders
create_folder(paths.task1_proc_train_data_folder)
create_folder(paths.task1_models_folder)
create_folder(paths.task1_proc_test_data_folder)
create_folder(paths.task1_test_res)
create_folder(paths.task2_train_folder)
create_folder(paths.task2_test_folder)

# define script actions
retrain_SETE_task1_models = False # If true: if models were already written to files, load them. else, train and write to files. If false: train and write models to files.
retrain_SETE_task2_model = False # If true: if model was already written to file, load it. else, train and write to file. If false: train and write model to file.
test_task_1 = True
test_task_2 = True

######## </editor-fold>

######## --------------- Get data for SETE ------------- ######## <editor-fold>

# change input data column names to suit the SETE format
rename_cols = {'TRB_CDR3': 'cdr3b',
               'TRBV': 'vb_gene'}

# Get the training data for each epitope from the separate files, write to new files after processing
# and also get the unified dataframe that contains data for all epitopes combined.
ep_dfs, df_all, epitopes = get_train_data(paths.input_training_data_folder, drop_duplicates_within_ep=True,
                                          duplicates_within_ep_cols=['TRB_CDR3', 'epitope'],
                                          drop_duplicates_between_all=False,
                                          duplicates_between_all_cols=['TRB_CDR3'],
                                          save_to_folder=paths.task1_proc_train_data_folder,
                                          rename_cols=rename_cols)

######## </editor-fold>

######## --------------- SETE Task 1 - separate binary predictors ------------- ######## <editor-fold>

if test_task_1:

    for epitope in epitopes:
        print(f'\n###### --------------- Epitope {epitope} --------------- ######')

        try:
            ######## --------------- get / train model ------------- ######## <editor-fold>

            # paths
            paths_ep = Object()
            paths_ep.training_data_filepname = os.path.join(paths.task1_proc_train_data_folder, f'epitope_{epitope}.csv')

            # epitope's output folder and file names
            paths_ep.epitope_model_folder = os.path.join(paths.task1_models_folder, epitope)
            paths_ep.epitope_model_filename = os.path.join(paths_ep.epitope_model_folder, epitope + '_model')
            paths_ep.epitope_preds_filename = os.path.join(paths_ep.epitope_model_folder, epitope + '_train_preds')
            paths_ep.epitope_kmers_filename = os.path.join(paths_ep.epitope_model_folder, epitope + '_train_kmers')
            paths_ep.epitope_PCA_filename = os.path.join(paths_ep.epitope_model_folder, epitope + '_train_PCA')
            paths_ep.epiname_list_train_filename = os.path.join(paths_ep.epitope_model_folder, epitope + '_epiname_list_train')
            paths_ep.input_test_ep_filename_orig = os.path.join(paths.input_test_data_folder, epitope + '.txt')
            paths_ep.input_test_ep_filename_csv = os.path.join(paths.task1_proc_test_data_folder, epitope + '.csv')
            paths_ep.epitope_test_res_filename = os.path.join(paths.task1_test_res, epitope + '.txt')
            create_folder(paths_ep.epitope_model_folder)

            if retrain_SETE_task1_models or not os.path.exists(paths_ep.epitope_model_filename): # if asked to retrain model or model file does not exist

                train_start_time = time.time()

                res = SETE.data_preprocess(paths_ep.training_data_filepname, 3,
                                           return_kmers=True, min_tcrs_amount=1)
                x_train_no_pca, y_train, epiname_list_train, kmers_list_train = res

                pca = PCA(n_components=0.9).fit(x_train_no_pca) # train PCA
                x_train = pca.transform(x_train_no_pca)

                # construct and train Gradient Boosting Classifier
                model = GradientBoostingClassifier(learning_rate=0.1,
                                                   min_samples_leaf=20, max_features='sqrt', subsample=0.8,
                                                   n_estimators=70, max_depth=11,
                                                   random_state=666,
                                                   min_samples_split=60,
                                                   loss="deviance" # LIEL this is basically log-loss
                                                   )

                model.fit(x_train, y_train)
                print(f'Epitope {epitope} training time:', get_str_time_from_start(train_start_time, sec_or_minute='minute'))

                # save model to file
                with open(paths_ep.epitope_model_filename, 'wb') as f:
                    pickle.dump(model, f)

                # save kmers to file
                with open(paths_ep.epitope_kmers_filename, 'wb') as f:
                    pickle.dump(kmers_list_train, f)

                # save PCA to file
                with open(paths_ep.epitope_PCA_filename, 'wb') as f:
                    pickle.dump(pca, f)

                # save epiname_list_train to file
                with open(paths_ep.epiname_list_train_filename, 'wb') as f:
                    pickle.dump(epiname_list_train, f)

            else:
                # get model from file
                with open(paths_ep.epitope_model_filename, 'rb') as f:
                    model = pickle.load(f)

                # get kmers from file
                with open(paths_ep.epitope_kmers_filename, 'rb') as f:
                    kmers_list_train = pickle.load(f)

                # get pca from file
                with open(paths_ep.epitope_PCA_filename, 'rb') as f:
                    pca = pickle.load(f)

                # get epiname_list_train from file
                with open(paths_ep.epiname_list_train_filename, 'rb') as f:
                    epiname_list_train = pickle.load(f)

            ######## </editor-fold>

            ######## --------------- test ------------- ######## <editor-fold>

            if test_task_1:
                print('### Testing model! Task 1')
                # test file name and import
                ep_test_df_orig = pd.read_csv(paths_ep.input_test_ep_filename_orig, sep='\t')
                print(f'Test data size: {ep_test_df_orig.shape[0]} samples')

                # write orig test file to csv
                ep_test_df = ep_test_df_orig.copy().rename(columns=rename_cols)
                ep_test_df['epitope'] = 'unknown'
                ep_test_df.to_csv(paths_ep.input_test_ep_filename_csv)

                # get prediction for test data
                res = SETE.data_preprocess(paths_ep.input_test_ep_filename_csv, 3,
                                           return_kmers=True, min_tcrs_amount=1)
                x_test_no_pca, y_test, epiname_list_test, kmers_list_test = res

                x_test_no_pca_df = pd.DataFrame(x_test_no_pca, columns=kmers_list_test)

                # Add 0s column for each kmer that is in the train set but not in the test set
                for kmer_train in kmers_list_train:
                    if kmer_train not in kmers_list_test:
                        x_test_no_pca_df[kmer_train] = 0

                # Remove kmers that are not in the train set, and reorder columns to match training set order
                x_test_no_pca_df__final = x_test_no_pca_df.loc[:, kmers_list_train]
                assert x_test_no_pca_df__final.shape[1] == len(kmers_list_train)

                # apply PCA over test set
                x_test = pca.transform(x_test_no_pca_df__final)

                # predict test set labels using the trained model
                y_test_preds_2 = model.predict_proba(x_test)

                if epiname_list_train[0] == epitope:
                    y_test_preds = y_test_preds_2[:, 0]
                elif epiname_list_train[1] == epitope:
                    y_test_preds = y_test_preds_2[:, 1]
                else:
                    raise Exception

                ep_test_df_orig['preds'] = y_test_preds

                print(f'Got {ep_test_df.shape[0] - np.isnan(ep_test_df_orig["preds"]).sum()} non-NA predictions out of {ep_test_df.shape[0]}')

                # save results to file
                ep_test_df_orig.to_csv(paths_ep.epitope_test_res_filename, sep='\t')

            ######## </editor-fold>

        except Exception:
            print(f'@@@@@@@@@ Couldnt get model / test for epitope {epitope}')

######## </editor-fold>

######## --------------- SETE Task 2 - single multiclass predictor ------------- ######## <editor-fold>

if test_task_2:

    paths_t2 = Object()

    # paths
    paths_t2.task2_train_data_path = os.path.join(paths.task2_train_folder, 'task2_training_data.csv')
    paths_t2.task2_model_path = os.path.join(paths.task2_train_folder, 'task2_model')
    paths_t2.task2_PCA_path = os.path.join(paths.task2_train_folder, 'task2_PCA')
    paths_t2.task2_kmers_path = os.path.join(paths.task2_train_folder, 'task2_train_kmers')
    paths_t2.task2_epi_map_path = os.path.join(paths.task2_train_folder, 'task2_epitope_mapping')
    paths_t2.task2_epiname_list_train_path = os.path.join(paths.task2_train_folder, 'task2_epiname_list_train')

    paths_t2.task2_test_data_path = os.path.join(paths.input_test_data_folder, 'testSet_Global_NoHeaders.txt')
    paths_t2.task2_test_data_path_csv = os.path.join(paths.task2_test_folder, 'testSet_Global_NoHeaders.csv')
    paths_t2.task2_test_res_path = os.path.join(paths.task2_test_folder, 'testSet_Global_NoHeaders_results.txt')
    paths_t2.task2_test_res_longform_path = os.path.join(paths.task2_test_folder, 'testSet_Global_NoHeaders_results_longform.csv')

    ######## --------------- SETE Task 2 - training ------------- ######## <editor-fold>

    print('Getting task2 model.')

    if retrain_SETE_task2_model or not os.path.exists(paths_t2.task2_model_path): # if asked to retrain model or model file does not exist

        # Get a unified training set - only positive examples of all epitopes
        df_train_task2 = df_all[df_all['epitope'] != 'none']
        df_train_task2.to_csv(paths_t2.task2_train_data_path)
        df_train_task2.epitope.value_counts().to_csv(os.path.join(paths.task2_train_folder, 'task2_training_data_summary.csv'))

        # preprocess training data and train model
        res = SETE.data_preprocess(paths_t2.task2_train_data_path, 3,
                                   return_kmers=True, min_tcrs_amount=1)
        x_train_no_pca, y_train, epiname_list_train, kmers_list_train = res
        epitope_labels_map = {i: epiname_list_train[i] for i in range(len(epiname_list_train))}

        # train PCA
        pca = PCA(n_components=0.9).fit(x_train_no_pca)
        x_train = pca.transform(x_train_no_pca)

        model = GradientBoostingClassifier(learning_rate=0.1,
                                           min_samples_leaf=20, max_features='sqrt', subsample=0.8,
                                           n_estimators=70, max_depth=11,
                                           random_state=666,
                                           min_samples_split=60,
                                           loss="deviance" # LIEL this is basically log-loss
                                           )

        # train Gradient Boosting Classifier
        train_start_time = time.time()
        model.fit(x_train, y_train)
        print(f'Task2 training time:', get_str_time_from_start(train_start_time, sec_or_minute='minute'))

        # save model to file
        with open(paths_t2.task2_model_path, 'wb') as f:
            pickle.dump(model, f)

        # save kmers to file
        with open(paths_t2.task2_kmers_path, 'wb') as f:
            pickle.dump(kmers_list_train, f)

        # save epitope_mapping to file
        with open(paths_t2.task2_epi_map_path, 'wb') as f:
            pickle.dump(epitope_labels_map, f)

        # save PCA model to file
        with open(paths_t2.task2_PCA_path, 'wb') as f:
            pickle.dump(pca, f)

        # save epiname_list_train model to file
        with open(paths_t2.task2_epiname_list_train_path, 'wb') as f:
            pickle.dump(epiname_list_train, f)

    else:
        # get model from file
        with open(paths_t2.task2_model_path, 'rb') as f:
            model = pickle.load(f)

        # get kmers from file
        with open(paths_t2.task2_kmers_path, 'rb') as f:
            kmers_list_train = pickle.load(f)

        # get epitope_labels_map from file
        with open(paths_t2.task2_epi_map_path,'rb') as f:
            epitope_labels_map = pickle.load(f)

        # get pca from file
        with open(paths_t2.task2_PCA_path, 'rb') as f:
            pca = pickle.load(f)

        # get epiname_list_train model from file
        with open(paths_t2.task2_epiname_list_train_path, 'rb') as f:
            epiname_list_train = pickle.load(f)

    ######## </editor-fold>

    ######## --------------- SETE Task 2 - testing ------------- ######## <editor-fold>

    print('Testing task2 model.')

    # read test data, rename columns and save to a new file
    df_test_task2_orig = pd.read_csv(paths_t2.task2_test_data_path, sep='\t')

    df_test_task2 = df_test_task2_orig.rename(columns=rename_cols)
    df_test_task2['epitope'] = 'unknown'
    df_test_task2.to_csv(paths_t2.task2_test_data_path_csv)

    # get prediction for test data
    res = SETE.data_preprocess(paths_t2.task2_test_data_path_csv, 3,
                               return_kmers=True, min_tcrs_amount=1)
    x_test_no_pca, y_test, epiname_list_test, kmers_list_test = res

    x_test_no_pca_df = pd.DataFrame(x_test_no_pca, columns=kmers_list_test)

    # Add 0s column for each kmer that is in the train set but not in the test set
    for kmer_train in kmers_list_train:
        if kmer_train not in kmers_list_test:
            x_test_no_pca_df[kmer_train] = 0

    # Remove kmers that are not in the train set, and reorder columns to match training matrix order
    x_test_no_pca_df__final = x_test_no_pca_df.loc[:, kmers_list_train]

    assert x_test_no_pca_df__final.shape[1] == len(kmers_list_train)

    x_test = pca.transform(x_test_no_pca_df__final)

    y_test_preds_labels = model.predict(x_test) # predicted label (highest scored epitope)
    y_test_preds = model.predict_proba(x_test) # predicted scores for all epitopes

    df_test_task2_res = df_test_task2_orig.copy()
    df_test_task2_res['preds_label'] = y_test_preds_labels
    df_test_task2_res['preds_epitope'] = df_test_task2_res['preds_label'].map(epitope_labels_map)

    print(f'Got {df_test_task2_res.shape[0] - np.isnan(df_test_task2_res["preds_label"]).sum()} non-NA predictions out of {df_test_task2_res.shape[0]}')

    df_test_task2_res.to_csv(paths_t2.task2_test_res_path, sep='\t')

    #### Reformat the task2 test results to a long form table with all epitopes scores
    df_test_task2_res_all_eps = pd.DataFrame(y_test_preds, columns=epiname_list_train)
    df_test_task2_res_all_eps = df_test_task2_orig.copy().join(df_test_task2_res_all_eps)
    df_test_task2_res_all_eps['TCR_id'] = df_test_task2_res_all_eps.index

    task2_test_long = df_test_task2_res_all_eps.melt(id_vars=['TCR_id']+list(df_test_task2_orig.columns),
                                                     value_name='score', var_name='predicted')
    task2_test_long = task2_test_long.sort_values('TCR_id')
    task2_test_long['rank'] = np.nan
    task2_test_long.reset_index(inplace=True, drop=True)

    for tcr_id in range(df_test_task2_orig.shape[0]):
        # rank epitope scores for a specific TCR
        tcr_df = task2_test_long.loc[task2_test_long['TCR_id'] == tcr_id]
        tcr_df['rank'] = tcr_df['score'].rank(method='min', ascending=False)
        # get these ranks to the main df
        task2_test_long.loc[task2_test_long['TCR_id'] == tcr_id, 'rank'] = tcr_df['rank']

    assert task2_test_long['rank'].isna().sum() == 0

    task2_test_long.to_csv(paths_t2.task2_test_res_longform_path, index=False)

    ######## </editor-fold>

######## </editor-fold>




