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
    print(mapper)


#fw.FileWiz

# Run main()
if __name__ == "__main__":
    main()
