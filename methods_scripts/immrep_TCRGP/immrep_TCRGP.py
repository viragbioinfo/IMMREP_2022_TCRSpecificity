"""
Train TCRGP over the ImmRep2022 training data, then test over test data
for task1 and task2. A binary classifier is trained for each epitope separately.
For task2, the binary models were used to predict the score for each epitope separately.
For each TCR, the epitopes were then ranked by their predicted scores, and the epitope
that got the highest score was chosen as the predicted epitope.

The script is assumed to be placed in the following directories tree:

python_projects/
    IMMREP_2022_TCRSpecificity/
        methods_scripts/
            immrep_TCRGP/
                immrep_TCRGP.py     <----------
    TCRGP/
        tcrgp.py
        ...
    ...

Script results will be written to the immrep_TCRGP folder (where the script is located).

Clone the TCRGP code from
"https://github.com/emmijokinen/TCRGP"
or, if my git pull request is not accepted yet, clone repository
"https://github.com/liel-cohen/TCRGP" instead. (the original emmijokinen code will not work with this script!)

Please notice that the cloned TCRGP folder should be placed in the root folder
(parent folder of the IMMREP_2022_TCRSpecificity folder).
If the TCRGP folder is located elsewhere, please change the tcrgp_package_folder variable in the
"Imports" part of the script, to the correct TCRGP folder string.

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
import pickle

import matplotlib.patches as patches
from matplotlib.patches import Patch

# add TCRGP project directory to paths so it will be importable
tcrgp_package_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))), 'TCRGP') # if TCRGP folder is in the parent folder. If not, set the path manually
sys.path.append(tcrgp_package_folder)
import tcrgp

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

def get_train_data(input_data_folder, drop_duplicates_within_ep=False,
                   duplicates_within_ep_cols=['TRB_CDR3', 'TRA_CDR3', 'TRAV', 'TRAJ', 'TRBV', 'TRBJ', 'epitope'],
                   drop_duplicates_between_all=False,
                   duplicates_between_all_cols=['TRB_CDR3', 'TRA_CDR3', 'TRAV', 'TRAJ', 'TRBV', 'TRBJ'],
                   save_to_folder=None):
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

            ep_dfs[ep] = ep_df
            num_rows_total += ep_df.shape[0]

            if save_to_folder is not None:
                ep_df.to_csv(os.path.join(save_to_folder, f'epitope_{ep}.csv'))

    df_all = pd.concat(list(ep_dfs.values()))
    assert df_all.shape[0] == num_rows_total

    ### Check for duplicates between all epitopes together
    df_all_pos = df_all.loc[df_all['Label'] == 1].copy()
    df_all_neg = df_all.loc[df_all['Label'] == -1].copy()

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

######## </editor-fold>

######## --------------- Params ------------- ######## <editor-fold>

# paths
paths = Object()

paths.TCRGP_folder = os.getcwd()
paths.main_immrep_folder = os.path.dirname(os.path.dirname(paths.TCRGP_folder))

# Data folders (in IMMREP_2022_TCRSpecificity folder)
paths.input_training_data_folder = os.path.join(paths.main_immrep_folder, 'training_data') # this folder should contain the files from https://github.com/viragbioinfo/IMMREP_2022_TCRSpecificity/tree/main/training_data
paths.input_test_data_folder = os.path.join(paths.main_immrep_folder, 'test_set') # this folder should contain the files from https://github.com/viragbioinfo/IMMREP_2022_TCRSpecificity/tree/main/test_set

# Output folders
paths.proc_train_data_folder = os.path.join(paths.TCRGP_folder, 'TCRGP_training_data_processed') # for writing the training data files in a TCRGP format
paths.models_folder = os.path.join(paths.TCRGP_folder, 'TCRGP_trained_models_TCRGP/')
paths.proc_test_data_folder = os.path.join(paths.TCRGP_folder, 'TCRGP_test_set_processed/') # for writing the test data files in a TCRGP format
paths.test_res_task1 = os.path.join(paths.TCRGP_folder, 'TCRGP__test_res_task1/') # output of the task1 test results
paths.test_res_task2 = os.path.join(paths.TCRGP_folder, 'TCRGP__test_res_task2/') # output of the task2 test results

create_folder(paths.proc_train_data_folder)
create_folder(paths.models_folder)
create_folder(paths.proc_test_data_folder)
create_folder(paths.test_res_task1)
create_folder(paths.test_res_task2)

# script actions
retrain_TCRGP_models = False # If true: if models were already written to files, load them. else, train and write to files. If false: train and write models to files.
test_task1 = True
test_task2 = True

######## </editor-fold>

######## --------------- Get training data for TCRGP ------------- ######## <editor-fold>

# Get the training data for each epitope from the separate files, write to new files after processing
# and also get the unified dataframe that contains data for all epitopes combined.
ep_dfs, df_all, epitopes = get_train_data(paths.input_training_data_folder, drop_duplicates_within_ep=True,
                                          duplicates_within_ep_cols=['TRB_CDR3', 'TRA_CDR3', 'TRAV', 'TRAJ', 'TRBV', 'TRBJ', 'epitope'],
                                          drop_duplicates_between_all=False,
                                          duplicates_between_all_cols=['TRB_CDR3', 'TRA_CDR3', 'TRAV', 'TRAJ', 'TRBV', 'TRBJ'],
                                          save_to_folder=paths.proc_train_data_folder)

######## </editor-fold>

######## --------------- TCRGP params ------------- ######## <editor-fold>

# Get BLOSUM62 matrix and its PCA projection matrix
subsmat = tcrgp.subsmatFromAA2('HENS920102', data_file=os.path.join(tcrgp_package_folder, 'data/aaindex2.txt'))
pc_blo = tcrgp.get_pcs(subsmat, d=21)

######## </editor-fold>

models = {}
thresholds = {}

for epitope in epitopes:
    print(f'\n###### --------------- Epitope {epitope} --------------- ######')

    ######## --------------- TCRGP train / get model ------------- ######## <editor-fold>

    ### paths
    ep_paths = Object()
    ep_paths.ep_training_data_filename = os.path.join(paths.proc_train_data_folder, f'epitope_{epitope}.csv')

    # epitope's output folder and file names
    ep_paths.epi_model_folder = os.path.join(paths.models_folder, epitope)
    ep_paths.epitope_model_filename = os.path.join(ep_paths.epi_model_folder, epitope + '_model')
    ep_paths.epitope_preds_filename = os.path.join(ep_paths.epi_model_folder, epitope + '_train_preds')
    ep_paths.epitope_auc_filename = os.path.join(ep_paths.epi_model_folder, epitope + '_train_auc.jpg')
    ep_paths.epitope_best_thresh_filename = os.path.join(ep_paths.epi_model_folder, epitope + '_train_best_threshold')
    create_folder(ep_paths.epi_model_folder)

    # train / get model
    if retrain_TCRGP_models or not os.path.exists(ep_paths.epitope_model_filename): # if asked to retrain model or model file does not exist
        train_start_time = time.time()

        train_df = pd.read_csv(ep_paths.ep_training_data_filename)
        max_cdr_len = max(train_df['TRB_CDR3'].apply(len).max(), train_df['TRA_CDR3'].apply(len).max())

        auc, params, preds = tcrgp.train_classifier(ep_paths.ep_training_data_filename,  # data file path
                                                    'human', epitope, pc_blo,
                                                    cdr_types=[['cdr3','cdr1','cdr2','cdr25'],['cdr3','cdr1','cdr2','cdr25']],
                                                    m_iters=500, lr=0.005, nZ=0, mbs=0,
                                                    lmax3=max_cdr_len,
                                                    balance_controls=False,
                                                    va='TRAV', vb='TRBV', cdr3a='TRA_CDR3', cdr3b='TRB_CDR3', epis='epitope',
                                                    alphabet_db_file_path=os.path.join(tcrgp_package_folder, 'data', 'alphabeta_db.tsv'),
                                                    return_preds=True)
        print(f'\nEpitope {epitope} training time:', get_str_time_from_start(train_start_time, sec_or_minute='minute'), '\n')

        tcrgp.print_model_info(params)

        # save model to file
        with open(ep_paths.epitope_model_filename,'wb') as f:
            pickle.dump(params, f)

        # save preds to file
        with open(ep_paths.epitope_preds_filename,'wb') as f:
            pickle.dump(preds, f)

        # plot ROC curves
        res = tcrgp.plot_aurocs_ths(params[7], preds, epi=epitope,
                              thresholds=[0.0, 0.05, 0.1, 0.2], dpi=500, figsize=(10, 3),
                              save_plot_path=ep_paths.epitope_auc_filename,
                              return_best_threshold=True)
        mean_auc, mean_wt_auc, auc_all, best_thresh = res
        print(f'{epitope}: training samples AUC: {auc}')

        # save best_thresh to file
        with open(ep_paths.epitope_best_thresh_filename,'wb') as f:
            pickle.dump(best_thresh, f)

        print('\n# Trained model and saved to file.\n')

    else:
        # get model from file
        with open(ep_paths.epitope_model_filename,'rb') as f:
            params = pickle.load(f)

        # get best threshold from file
        with open(ep_paths.epitope_best_thresh_filename,'rb') as f:
            best_thresh = pickle.load(f)

        print('# Loaded model from file.')

    # save model and thresholds to dictionaries, for task 2 testing
    models[epitope] = params
    thresholds[epitope] = best_thresh

    ######## </editor-fold>

    ######## --------------- TCRGP test (task 1 - binary models) ------------- ######## <editor-fold>

    if test_task1:
        print('\n### Testing model!')
        # test file name and import
        ep_paths.input_test_ep_filename_orig = os.path.join(paths.input_test_data_folder, epitope + '.txt')
        ep_test_df = pd.read_csv(ep_paths.input_test_ep_filename_orig, sep='\t')
        print(f'Test data size: {ep_test_df.shape[0]} samples')

        # write orig test file to csv (TCRGP format)
        ep_paths.input_test_ep_filename_csv = os.path.join(paths.proc_test_data_folder, epitope + '.csv')
        ep_test_df.to_csv(ep_paths.input_test_ep_filename_csv)

        # get prediction for test data
        preds = tcrgp.predict(ep_paths.input_test_ep_filename_csv, params, organism='human',
                              va='TRAV', vb='TRBV', cdr3a='TRA_CDR3', cdr3b='TRB_CDR3',
                              alphabet_db_file_path=os.path.join(tcrgp_package_folder, 'data', 'alphabeta_db.tsv'))
        print(f'\nGot {ep_test_df.shape[0] - np.isnan(preds).sum()} predictions, out of requested {ep_test_df.shape[0]}')

        ep_test_df['preds'] = preds

        # Write test results to file
        ep_test_df.to_csv(os.path.join(paths.test_res_task1, epitope + '.txt'), sep='\t')

    ######## </editor-fold>

######## --------------- TCRGP test - task 2 (combine binary models for multiclass prediction) ------------- ######## <editor-fold>

normalize_to_thresh = False # normalize predictions of each epitope to its model best threshold based on the training set

if test_task2:
    # paths
    paths.task2_test_data_filename = os.path.join(paths.input_test_data_folder, 'testSet_Global_NoHeaders.txt')
    paths.task2_test_data_filename_csv = os.path.join(paths.test_res_task2, 'testSet_Global_NoHeaders.csv')
    paths.task2_test_results_filename = os.path.join(paths.test_res_task2, 'testSet_Global_NoHeaders_results.txt')
    paths.task2_test_results_longform_filename = os.path.join(paths.test_res_task2, 'testSet_Global_NoHeaders_results_longform.csv')

    # read test file and save as csv
    df_test_task2_orig = pd.read_csv(paths.task2_test_data_filename, sep='\t')
    df_test_task2_orig.to_csv(paths.task2_test_data_filename_csv)

    # get predictions from each epitopes model and add to results df
    df_test_task2_res = df_test_task2_orig.copy()

    for epitope in epitopes:
        df_test_task2_res[epitope] = np.nan
        model = models[epitope]

        preds = tcrgp.predict(paths.task2_test_data_filename_csv, model, organism='human',
                              va='TRAV', vb='TRBV', cdr3a='TRA_CDR3', cdr3b='TRB_CDR3',
                              alphabet_db_file_path=os.path.join(tcrgp_package_folder, 'data', 'alphabeta_db.tsv'))
        df_test_task2_res[epitope] = preds

        if normalize_to_thresh:
            thresh = thresholds[epitope]
            df_test_task2_res[epitope] = (df_test_task2_res[epitope] / thresh) / (1/thresh)

        print(f'{epitope}: got {df_test_task2_res.shape[0] - np.isnan(preds).sum()} predictions, out of requested {df_test_task2_res.shape[0]}\n')

    #### Reformat the task2 test results to a long form table with all epitopes scores, and save to file
    df_test_task2_res['TCR_id'] = df_test_task2_res.index

    task2_test_long = df_test_task2_res.melt(id_vars=['TCR_id']+list(df_test_task2_orig.columns),
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

    assert task2_test_long['rank'].isna().sum() == df_test_task2_res[epitopes].isna().sum().sum()

    task2_test_long.to_csv(paths.task2_test_results_longform_filename, index=False)

######## </editor-fold>
