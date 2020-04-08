# -*- coding: utf-8 -*-
"""

==========
Created on 4/7/2020 at 9:55 PM
==========
Name: fileprep_classification.py
==========
Description: This script is deisgned to move files from their original folders to class-based folders.
==========

"""

# Library Imports
# Useful for this type of file preparation
import imagewiz.fw as fw

import pandas as pd
import os

# Static filepaths
path_to_map = os.path.join(os.getcwd(),
                           "projectfiles",
                           "rsna-pneumonia-detection-challenge",
                           "stage_2_detailed_class_info.csv")

path_to_train_files = os.path.join(os.getcwd(),
                                   "projectfiles",
                                   "rsna-pneumonia-detection-challenge",
                                   "stage_2_train_images")

# Define functions



# Define main() function
def main():
    # Import map file for classes
    mapper = pd.read_csv(path_to_map)

    # Create alias for classes
    mapper['class'].replace("No Lung Opacity / Not Normal", "2", inplace=True)
    mapper['class'].replace("Lung Opacity", "1", inplace=True)
    mapper['class'].replace("Normal", "0", inplace=True)

    # Prepare to manipulate the dicom files
    file_object = fw.FileWiz(mapper, path_to_train_files)
    file_object.classgrab("patientId", verbose=False)
    file_object.foldermove(os.path.join(os.getcwd(), "projectfiles", "rsna-pneumonia-detection-challenge"),
                           train_folder_name="train",
                           val_folder_name="validation",
                           tra_t_split=0.2,
                           extension_type=".dcm",
                           activate=True)


# Run main()
if __name__ == "__main__":
    main()
