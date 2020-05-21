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

import pydicom
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

def single_png_converter(path_to_png, destination_filepath):
    ds = pydicom.dcmread(os.path.join(path + filename))
    shape = ds.pixel_array.shape
    # Convert to float to avoid overflow or underflow losses.
    image_2d = ds.pixel_array.astype(float)

    # Rescaling grey scale between 0-255
    image_2d_scaled = (np.maximum(image_2d, 0) / image_2d.max()) * 255.0

    # Convert to uint
    image_2d_scaled = np.uint8(image_2d_scaled)

    # defines the file type as png instead of dcm
    p_filename = filename[:-3] + "png"

    with open(os.path.join(dest_folder + p_filename), 'wb') as png_file:
        w = png.Writer(shape[1], shape[0], greyscale=True)
        w.write(png_file, image_2d_scaled)

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
