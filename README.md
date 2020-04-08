# Kaggle Pneumonia Detection Challenge
A folder for the [RSNA Pneumonia Prediction Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)

Authors: [Evan Gibson](https://github.com/evangibson), [Derek Turner](https://github.com/turnerdj95), and [Thomas Nakrosis](https://github.com/Tnakrosis).

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

### Metrics for Success  
---
#### Phase 1 Metrics
The first phase of this project will deal with separating each image into one of three classes: healthy, pneumonia, and non-healthy/non-pneumonia.  

For this multiclass classification task, there are a few metrics that can measure just how successful our machine learning model is.  
These include: 

> **Accuracy:**  The number of correct predictions divided by the number of all samples.  This can be deduced by the following formula: 
>  
> <img src="https://render.githubusercontent.com/render/math?math=Accuracy = \frac{TP%2BTN}{TP%2BTN%2BFP%2BFN}">
> 
> **Precision:**  Also known as *positive predictive value (PPV)*, measures how many of the samples predicted as positive are *actually* positive.  This is best used as a performance metric when the goal is to limit the number of false positives.  This can be deduced by the following formula:  
>   
> <img src="https://render.githubusercontent.com/render/math?math=Precision = \frac{TP}{TP%2BFP}">
>  
> **Recall:**  Also known as *sensitivity, hit rate, or true positive rate (TPR)*, measures how many of the positive samples are captured by the positive predictions.  This is best used when there is a need to identify all positive samples; that is, when it is important to avoid false negatives.  This can be deduced by the following formula:  
>  
> <img src="https://render.githubusercontent.com/render/math?math=Recall = \frac{TP}{TP%2BFN}">
>  
> **F-score:**  While precision and recall are very important measures, looking at only one of them won't provide the full picture of our data.  One way to summarize the two is with the *f-score*, which is the harmonic mean of precision and recall:  
>  
> <img src="https://render.githubusercontent.com/render/math?math=F = 2 \times \frac{Precision \times Recall}{Precision%2BRecall}">
>  
> *TP: True Positives, TN: True Negatives, FP: False Positives, FN: False Negatives*  

For our purposes, the F-score is our most important metric.  If the purpose of this project is to minimize the workload of medical staff, it would be counter-intuitive to allow too many false positives, as this will prompt follow-on analysis from the staff; thus, we will need to ensure a high Precision score.  However, with our model circling around an infection that has such dire consequences if not treated, false negatives cannot be permitted; thus, we will also need a high recall score.  

It will be important to look at both precision and recall individually, but the F-score will be a neat summary of the two that we can use to standardize our progress.

---
#### Phase 2 Metrics
The second phase of this project will involve identifying the "trouble areas" of those lungs placed in the pneumonia class.  

_To be completed_

---
#### Key Metrics from the Bounding Box Detector *(Bounding Box Predictor (CNN).ipynb)*
🔧 _Image incoming_ 🔧

#### Key Metrics from a Binary Classifier *(Binary Classifier 2.ipynb)*

🔧 _Image incoming_ 🔧

It's pretty clear that, for now, the Bounding Box Detector *does not* need any help from my current classifiers. 

___
## Acknowledgments and References
