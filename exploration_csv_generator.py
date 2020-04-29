"""

==========
Created on 4/29/2020 at 5:00 PM
==========
Name: exploration_csv_generator.py
==========
Description: This script is designed to create a new csv file by reading in DICOM files, pulling out necessary data,
and appending it to a mass dataframe that will be saved for follow-on analysis.
==========

"""

import pandas as pd
import numpy as np
import pydicom, os, tqdm

# Declare static filepath:
detailed_path = r'H:\rsna-pneumonia-detection-challenge\stage_2_detailed_class_info.csv'
base_filepath = r'H:\rsna-pneumonia-detection-challenge\stage_2_train_images'

# Read in csv:


# Simplify reading in .dcm files:
def read_dcm(pid, dataframe, base_path=base_filepath):
    deets = dataframe[dataframe['patientId'] == pid]
    path = os.path.join(base_path, deets['class'].iloc[0], pid + '.dcm')
    dcm_data = pydicom.read_file(path)
    return dcm_data


# Binary column function
def binary_classifier(dataframe):
    unhealthy_df = dataframe[dataframe['class'] == 'class_2']
    unhealthy_df['bin'] = 1
    new_df = pd.merge(dataframe, unhealthy_df, how='left', on=['patientId', 'class'])
    new_df['bin'].fillna(0, inplace=True)
    new_df['bin'] = new_df['bin'].astype(int)
    return new_df


def info_grab(pid, keyword, dataframe):
    temp = read_dcm(pid, dataframe)
    value = temp[keyword][:]
    return value


def col_gen(dataframe, keyword):
    dataframe[keyword] = np.nan
    for i in tqdm.tqdm(dataframe.index):
        dataframe.loc[i, keyword] = info_grab(dataframe.loc[i, 'patientId'], keyword, dataframe)
    return dataframe


def main():
    # Read in .csv's
    detailed_df = pd.read_csv(detailed_path)
    detailed_df.drop_duplicates(inplace=True)

    # Make adjustments
    detailed_df['class'].replace('No Lung Opacity / Not Normal', 'class_1', inplace=True)
    detailed_df['class'].replace('Normal', 'class_0', inplace=True)
    detailed_df['class'].replace('Lung Opacity', 'class_2', inplace=True)

    detailed_df = binary_classifier(detailed_df)
    detailed_df = col_gen(detailed_df, 'PatientAge')
    detailed_df = col_gen(detailed_df, 'PatientSex')

    detailed_df.to_csv("patient_info.csv", index=False)


if __name__ == '__main__':
    main()