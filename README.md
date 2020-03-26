# Kaggle Pneumonia Detection Challenge
A folder for the [RSNA Pneumonia Prediction Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)

Authors: [Evan Gibson](https://github.com/evangibson), [Derek Turner](https://github.com/turnerdj95), and Thomas Nakrosis.

___

## Table of Contents
- Summary
- File Descriptions
- Exploratory Analysis
- Model Results
- Acknowledgments and References

___
## Summary

For this contest, our team took two distinct approaches to the data; classifying the images into buckets and predicting bounding boxes. Presently, our approach hinges on the hypothesis that we can significantly improve bounding box predictions if we can create a classifier that identifies images with/without pneumonia. 

___
## Exploratory Analysis

It's important to understand the data in our DICOM files before we dive in. Here is an example of a method one can employ to view the contents of a dicom file:

```python
import pydicom
dcm_file = path/to/dicom/file.dcm
dcm_data = pydicom.read_file(dcm_file)
print(dcm_data)
```
🔧 _Image incoming_ 🔧

Understanding the data from the DICOM files is imperative to being able to ensure one's conceptualization of bounding boxes on the arrays from those files. After all, our team does not have radiologists nor do its members have medical training. Therefore, we need to visualize those boxes in order to augment our knowledge regarding the visual aspects of pneumonia:

🔧 _Image incoming_ 🔧

Now that we have visualized the bounding boxes and the XRAYs, we should take a look at the demographics of our study participants.

#### Frequency Chart of Detailed Class (Colored by Binary Class)
🔧 _Image incoming_ 🔧

#### Frequency Chart of Binary Class (Colored by Binary Class)

🔧 _Image incoming_ 🔧

#### Frequency Chart of Sex (Colored by Binary Class)

🔧 _Image incoming_ 🔧

Ensuring that we maintain reasonably consistent demographic spreads when determining training and test sets will be imperative. Exploratory analysis will assist in that effort.

Making sure we understanding the pneumonia spread in these images will help us to identify anomalies in our predictions.

#### Heatmap of Pneumonia Presence in the Sample Images
🔧 _Image incoming_ 🔧

___
## Model Results

_Text blurb on model results_

#### Key Metrics from the Bounding Box Detector *(Bounding Box Predictor (CNN).ipynb)*
🔧 _Image incoming_ 🔧

#### Key Metrics from a Binary Classifier *(Binary Classifier 2.ipynb)*

🔧 _Image incoming_ 🔧

It's pretty clear that, for now, the Bounding Box Detector *does not* need any help from my current classifiers. 

___
## Acknowledgments and References