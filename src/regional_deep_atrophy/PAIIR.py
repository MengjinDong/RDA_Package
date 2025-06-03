import os
import random
import shutil
import numpy as np
import pandas as pd
import scipy.linalg
import pathlib
import argparse
from sklearn.linear_model import LinearRegression
from datetime import datetime
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json

pd.set_option('display.max_columns', None)

# Convert numpy arrays to lists
def convert_ndarray(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_ndarray(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray(item) for item in obj]
    else:
        return obj


# The main program launcher
class PAIIRLauncher:

    def __init__(self, parse):

        # Add the arguments
        parse.add_argument('--train-pair-spreadsheet', metavar='train_spreadsheet.csv',
                           type=pathlib.Path, 
                           default='',
                           help='filename of the train spreadsheet')
        parse.add_argument('--test-pair-spreadsheet', metavar='test_spreadsheet.csv', 
                           type=pathlib.Path, 
                           default='',
                           help=' filename of the test spreadsheet')
        parse.add_argument('--test-double-pair-spreadsheet', metavar='spreadsheet to predict STO', 
                           type=pathlib.Path, 
                           default='',
                           help='filename of the spreadsheet for STO prediction')
        parse.add_argument('--prefix', metavar='prefix', type=str, 
                           default='model_',
                           help='Prefix for the output files. Default is None.')
        parse.add_argument('--workdir', type=str, metavar='/path/to/workdir',
                           default='',
                           help='Location to store intermediate files. Default is system temp directory.')
        parse.add_argument('--debug', action='store_true', 
                           help='Enable verbose/debug mode. Default is False.')
        parse.add_argument('--min-date', default=180, type=int, metavar='N1',
                           help='Minimum date difference to consider. Default is 180 days.')
        parse.add_argument('--max-date', default=400, type=int, metavar='N2',
                           help='Minimum date difference to consider. Default is 400 days.')
        

        # Set the function to run
        parse.set_defaults(func = lambda args : self.run(args))

    def run(self, args):

        self.args = args 

        # preprocessing, read spreadsheet
        self.read_and_organize()

        # STO accuracy
        self.STO_accuracy()

        # RISI accuracy
        self.RISI_accuracy()

        # obtain a single measurement for each subject (predicted interscan interval, and PAIIR)
        self.obtain_PAIIR()

        # plot PAIIR for each stage, not corrected for age
        self.plot_PAIIR()


    def read_and_organize(self):
        # from the last five nodes of the model prediction, 
        # obtain the predicted interscan interval for each scan pair

        # Read CSV files
        self.train_pair = pd.read_csv(self.args.train_pair_spreadsheet)
        self.train_pair = self.train_pair.rename(columns={
            'bl_time1': 'bl_time',
            'fu_time1': 'fu_time',
            'date_diff1': 'date_diff_true',
            'label_date_diff1': 'label_date_diff',
            'pred_date_diff1': 'pred_date_diff'
        })

        # Subset the data for stage == 0
        train_spreadsheet0 = self.train_pair[self.train_pair['stage'] == 0]

        # Load the test pair CSV
        self.test_pair = pd.read_csv(self.args.test_pair_spreadsheet)

        # Rename columns in the test pair
        self.test_pair = self.test_pair.rename(columns={
            'bl_time1': 'bl_time',
            'fu_time1': 'fu_time',
            'date_diff1': 'date_diff_true',
            'label_date_diff1': 'label_date_diff',
            'pred_date_diff1': 'pred_date_diff'
        })

        # Fit the linear model
        X_train = train_spreadsheet0[['score0', 'score1', 'score2', 'score3', 'score4']]
        y_train = train_spreadsheet0['date_diff_true']
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict on the test pair
        X_test = self.test_pair[['score0', 'score1', 'score2', 'score3', 'score4']]
        self.test_pair['CNN_date_diff'] = model.predict(X_test)

        # Convert date columns to datetime format
        self.test_pair['bl_time'] = pd.to_datetime(self.test_pair['bl_time']) # , format='%Y-%m-%d'
        self.test_pair['fu_time'] = pd.to_datetime(self.test_pair['fu_time'])
        self.train_pair['bl_time'] = pd.to_datetime(self.train_pair['bl_time'])
        self.train_pair['fu_time'] = pd.to_datetime(self.train_pair['fu_time'])

        # Now test_spreadsheet and train_spreadsheet have the date columns in proper format and test_spreadsheet has predictions.
        self.test_double_pair = pd.read_csv(self.args.test_double_pair_spreadsheet)

        # print column names
        # column_names = self.test_double_pair.columns
        # print("column names of double pair spreadsheet", column_names)
        # print("double pair spreadsheet")
        # print(self.test_double_pair.head())

        return 
        

    def STO_accuracy(self):

        # Assuming common_subj_1 is already a pandas DataFrame

        # column_names = self.test_pair.columns
        # print("column names of single pair spreadsheet", column_names)
        # print("single pair spreadsheet")
        # print(self.test_pair.head())

        # Mutate to create CNN_pred_date_diff
        self.test_pair['CNN_pred_date_diff'] = self.test_pair['CNN_date_diff'].apply(lambda x: 1 if x > 0 else 0)

        # Convert columns to categorical (factor equivalent in R)
        self.test_pair['CNN_pred_date_diff'] = self.test_pair['CNN_pred_date_diff'].astype(int).astype('category')
        self.test_pair['label_date_diff'] = self.test_pair['label_date_diff'].astype(int).astype('category')

        # Accuracy for CNN by group (stage)
        cnn_accuracy_by_group = self.test_pair.groupby('stage').apply(
            lambda group: (group['label_date_diff'] == group['CNN_pred_date_diff']).sum() / len(group)
        ).reset_index(name='accuracy')

        # Overall accuracy for CNN
        cnn_overall_accuracy = (self.test_pair['label_date_diff'] == self.test_pair['CNN_pred_date_diff']).mean()

        # Print results
        print("CNN STO accuracy by group:")
        print(cnn_accuracy_by_group)

        print("\nOverall CNN STO accuracy:", cnn_overall_accuracy)

        cnn_confusion_matrix = confusion_matrix(self.test_pair['label_date_diff'], self.test_pair['CNN_pred_date_diff'])
        
        print("\nCNN STO Confusion Matrix:")
        print(cnn_confusion_matrix)
        
        cnn_accuracy_by_group_dict = cnn_accuracy_by_group.to_dict(orient='records') 
        cnn_confusion_matrix_serializable = convert_ndarray(cnn_confusion_matrix)

        self.accuracies = {
            'STO_accuracy_by_group': cnn_accuracy_by_group_dict,
            'STO_overall_accuracy': cnn_overall_accuracy,
            'STO_confusion_matrix': cnn_confusion_matrix_serializable
        }

        with open(self.args.workdir + '/accuracy.json', 'w') as json_file:
            json.dump(self.accuracies, json_file, indent=4)

        return

    def RISI_accuracy(self):

        self.test_pair = self.test_pair.rename(columns={
            'bl_time': 'bl_time1',
            'fu_time': 'fu_time1'
        })

        self.test_double_pair['side'] = self.test_double_pair['side'].astype(str)
        self.test_pair['side'] = self.test_pair['side'].astype(str)

        self.test_double_pair['stage'] = self.test_double_pair['stage'].astype(int)
        self.test_pair['stage'] = self.test_pair['stage'].astype(int)

        self.test_double_pair['bl_time1'] = pd.to_datetime(self.test_double_pair['bl_time1'], format="%m/%d/%y %H:%M")
        self.test_double_pair['fu_time1'] = pd.to_datetime(self.test_double_pair['fu_time1'], format="%m/%d/%y %H:%M")

        self.test_double_pair['bl_time2'] = pd.to_datetime(self.test_double_pair['bl_time2'], format="%m/%d/%y %H:%M")
        self.test_double_pair['fu_time2'] = pd.to_datetime(self.test_double_pair['fu_time2'], format="%m/%d/%y %H:%M")


        self.test_pair['bl_time1'] = pd.to_datetime(self.test_pair['bl_time1'], format="%m/%d/%y %H:%M")
        self.test_pair['fu_time1'] = pd.to_datetime(self.test_pair['fu_time1'], format="%m/%d/%y %H:%M")

        test_double_pair1 = self.test_double_pair.merge(self.test_pair, on=["subjectID", "side", "stage", "bl_time1", "fu_time1"]) \
                                    .rename(columns={'CNN_date_diff': 'CNN_date_diff1'})

        # rename bl_time1 to bl_time2, fu_time1 to fu_time2
        self.test_pair = self.test_pair.rename(columns={
            'bl_time1': 'bl_time2',
            'fu_time1': 'fu_time2'
        })

        # match from test_pair so that both pairs are in test_double_pair
        test_double_pair2 = test_double_pair1.merge(self.test_pair, on=["subjectID", "side", "stage", "bl_time2", "fu_time2"]) \
                                .rename(columns={'CNN_date_diff': 'CNN_date_diff2'}) \
                                .assign(CNN_date_diff_ratio=lambda df: df.apply(
                                    lambda row: 1 if abs(row['CNN_date_diff2']) > abs(row['CNN_date_diff1']) else 0, axis=1)) \
                                .assign(label_time_interval_binary=lambda df: df.apply(
                                    lambda row: 0 if row['label_time_interval'] >= 2  else 1, axis=1))

        self.test_pair = self.test_pair.rename(columns={
            'bl_time2': 'bl_time',
            'fu_time2': 'fu_time'
        })

        # Group by 'stage' and calculate accuracy for CNN by group
        cnn_accuracy_by_group = test_double_pair2.groupby('stage').apply(
            lambda group: (group['label_time_interval_binary'] == group['CNN_date_diff_ratio']).sum() / len(group)
        ).reset_index(name='accuracy')

        # Overall accuracy for CNN
        cnn_overall_accuracy = (test_double_pair2['label_time_interval_binary'] == test_double_pair2['CNN_date_diff_ratio']).mean()

        # Convert label_time_interval and CNN_date_diff_ratio to categorical
        test_double_pair2['label_time_interval_binary'] = test_double_pair2['label_time_interval_binary'].astype('category')
        test_double_pair2['CNN_date_diff_ratio'] = test_double_pair2['CNN_date_diff_ratio'].astype('category')

        # Confusion matrix for CNN
        cnn_confusion_matrix = confusion_matrix(test_double_pair2['label_time_interval_binary'], test_double_pair2['CNN_date_diff_ratio'])

        # Print results
        print("CNN RISI accuracy by group:")
        print(cnn_accuracy_by_group)

        print("\nOverall CNN RISI accuracy:", cnn_overall_accuracy)

        print("\nCNN RISI Confusion Matrix:")
        print(cnn_confusion_matrix)

        cnn_accuracy_by_group_dict = cnn_accuracy_by_group.to_dict(orient='records') 
        cnn_confusion_matrix_serializable = convert_ndarray(cnn_confusion_matrix)

        self.accuracies['RISI_accuracy_by_group'] = cnn_accuracy_by_group_dict
        self.accuracies['RISI_overall_accuracy'] = cnn_overall_accuracy
        self.accuracies['RISI_confusion_matrix'] = cnn_confusion_matrix_serializable

        with open(self.args.workdir + '/accuracy.json', 'w') as json_file:
            json.dump(self.accuracies, json_file, indent=4)  
        
        print("\n manual calculation of confusion matrix")

        test_double_pair2.to_csv(self.args.workdir + '/test_double_pair2.csv', index=False)
        
        # Iterate through each unique stage
        for stage in test_double_pair2['stage'].unique():
            # Filter DataFrame for the current stage
            test_double_pair_temp_stage = test_double_pair2[test_double_pair2['stage'] == stage]

            count1 = test_double_pair_temp_stage[(test_double_pair_temp_stage['label_time_interval_binary'] == 0) & (test_double_pair_temp_stage['CNN_date_diff_ratio'] == 0)].shape[0]
            count2 = test_double_pair_temp_stage[(test_double_pair_temp_stage['label_time_interval_binary'] == 0) & (test_double_pair_temp_stage['CNN_date_diff_ratio'] == 1)].shape[0]
            count3 = test_double_pair_temp_stage[(test_double_pair_temp_stage['label_time_interval_binary'] == 1) & (test_double_pair_temp_stage['CNN_date_diff_ratio'] == 0)].shape[0]
            count4 = test_double_pair_temp_stage[(test_double_pair_temp_stage['label_time_interval_binary'] == 1) & (test_double_pair_temp_stage['CNN_date_diff_ratio'] == 1)].shape[0]

            print("RISI confusion matrix for stage ", stage, " is:")

            print("count1", count1)
            print("count2", count2)
            print("count3", count3)
            print("count4", count4)

        return


    def obtain_PAIIR(self):

        common_subj_2 = self.test_pair.copy()

        # save csv file
        common_subj_2.to_csv(self.args.workdir + '/common_subj_2.csv', index=False)

        # Update columns conditionally based on the value of date_diff_true
        common_subj_2["bl_time"] = common_subj_2.apply(
            lambda row: row["bl_time"] if row["date_diff_true"] > 0 else row["fu_time"], axis=1
        )
        common_subj_2["fu_time"] = common_subj_2.apply(
            lambda row: row["fu_time"] if row["date_diff_true"] > 0 else row["bl_time"], axis=1
        )
        common_subj_2["CNN_date_diff"] = common_subj_2.apply(
            lambda row: row["CNN_date_diff"] if row["date_diff_true"] > 0 else -row["CNN_date_diff"], axis=1
        )
        common_subj_2["date_diff_true"] = common_subj_2.apply(
            lambda row: row["date_diff_true"] if row["date_diff_true"] > 0 else -row["date_diff_true"], axis=1
        )

        # Step 3: Aggregating by specific columns and computing the mean
        common_subj_3 = common_subj_2.groupby(["subjectID", "stage", "bl_time", "fu_time"]).agg({
            "CNN_date_diff": "mean",
            "date_diff_true": "mean",
        }).reset_index()

        # Final result
        common_subj = common_subj_3

        self.obtain_PAIIR_per_subject(common_subj)

        return


    def obtain_PAIIR_per_subject(self, common_subj):
        
        common_subj_zoom_in = common_subj[(common_subj['date_diff_true'] >= self.args.min_date) & (common_subj['date_diff_true'] < self.args.max_date)]

        common_subj_zoom_in['CNN_date_diff_ratio'] = common_subj_zoom_in['CNN_date_diff'] / common_subj_zoom_in['date_diff_true']


        # Calculate weighted sum for the numerator
        weighted_sum_nume_CNN = common_subj_zoom_in.groupby("subjectID").apply(
            lambda x: (x['CNN_date_diff_ratio'] * x['date_diff_true'] * 365).sum()
        ).reset_index(name="CNN_weighted_sum_nume")

        # Calculate weighted sum for the denominator
        weighted_sum_deno = common_subj_zoom_in.groupby("subjectID").apply(
            lambda x: (x['date_diff_true'] * x['date_diff_true']).sum()
        ).reset_index(name="weighted_sum_deno")

        # Join the weighted sums and add distinct fields for each subjectID
        self.weighted_atrophy = (
            weighted_sum_nume_CNN
            .merge(weighted_sum_deno, on="subjectID")
            .merge(common_subj_zoom_in.drop_duplicates("subjectID"), on="subjectID")
        )

        # Calculate the DA_Atrophy_raw column
        self.weighted_atrophy['DA_Atrophy'] = (
            self.weighted_atrophy['CNN_weighted_sum_nume'] / self.weighted_atrophy['weighted_sum_deno']
        )

        # Calculate the PAIIR column
        print("weighted_atrophy.head()")
        print(self.weighted_atrophy.head())

        self.weighted_atrophy.to_csv(self.args.workdir + '/weighted_atrophy.csv', index=False)

        return

    def plot_PAIIR(self):
        
        # Grouping by stage and calculating the mean of weighted_atrophy

        stage_labels = {0: "A- CU", 1: "A+ CU", 3: "A+ eMCI", 5: "A+ lMCI"}
        stage_order = ["A- CU", "A+ CU", "A+ eMCI", "A+ lMCI"]

        self.weighted_atrophy['stage'] = self.weighted_atrophy['stage'].map(stage_labels)

        mean_se = self.weighted_atrophy.groupby('stage')['DA_Atrophy'].agg(
            mean='mean', 
            se=lambda x: np.std(x, ddof=1) / np.sqrt(len(x))  # Standard error
        ).reindex(stage_order).reset_index()

        print("mean_se", mean_se)

        # Create a bar plot
        plt.figure(figsize=(8, 6))
        sns.set_palette("Paired")  # Brighter color palette

        bar_plot = sns.barplot(x='stage', y='mean', data=mean_se, order=stage_order, ci=None)


        plt.errorbar(x=range(len(mean_se)), y=mean_se['mean'], yerr=mean_se['se'], fmt='none', c='black', capsize=5)

        # Perform t-tests for each group against "A- CU" and display p-values
        base_group = self.weighted_atrophy[self.weighted_atrophy['stage'] == "A- CU"]['DA_Atrophy']
        for idx, stage in enumerate(stage_order[1:]):  # Skip the base group itself
            comparison_group = self.weighted_atrophy[self.weighted_atrophy['stage'] == stage]['DA_Atrophy']
            _, p_value = stats.ttest_ind(base_group, comparison_group)
            
            # Format p-value in scientific notation
            p_text = f"p = {p_value:.1e}"
            bar_plot.text(idx + 1, mean_se['mean'][idx + 1] + mean_se['se'][idx + 1] * 1.1, p_text, ha='center', color='black')

        # Add titles and labels
        plt.title('Mean Weighted PAIIR by Diagnosis (Comparing with A- CU)')
        plt.xlabel('Diagnosis')
        plt.ylabel('Weighted PAIIR (mean ± SE)')
        plt.xticks(ticks=range(len(stage_order)), labels=stage_order, rotation=45)
        # plt.ylim(0, max(mean_values['DA_Atrophy']) + 0.1)  # Adjust y limit for p-value display

        # Show the plot
        plt.show()

        plt.savefig(self.args.workdir + '/PAIIR_group_diff.png')

