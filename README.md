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

It's important to understand the data in our DICOM (.dcm) files before we dive in. Here is an example of a method one can employ to view the contents of a DICOM file:

```python
import pydicom
dcm_file = path/to/dicom/file.dcm
dcm_data = pydicom.read_file(dcm_file)
print(dcm_data)
```
```
(0008, 0005) Specific Character Set              CS: 'ISO_IR 100'
(0008, 0016) SOP Class UID                       UI: Secondary Capture Image Storage\n",
(0008, 0018) SOP Instance UID                    UI: 1.2.276.0.7230010.3.1.4.8323329.1556.1517874291.545552
(0008, 0020) Study Date                          DA: '19010101'
(0008, 0030) Study Time                          TM: '000000.00'
(0008, 0050) Accession Number                    SH: ''
(0008, 0060) Modality                            CS: 'CR'
(0008, 0064) Conversion Type                     CS: 'WSD'
(0008, 0090) Referring Physician's Name          PN: ''
(0008, 103e) Series Description                  LO: 'view: AP'
(0010, 0010) Patient's Name                      PN: '00f08de1-517e-4652-a04f-d1dc9ee48593'
(0010, 0020) Patient ID                          LO: '00f08de1-517e-4652-a04f-d1dc9ee48593'
(0010, 0030) Patient's Birth Date                DA: ''
(0010, 0040) Patient's Sex                       CS: 'M'
(0010, 1010) Patient's Age                       AS: '58'
(0018, 0015) Body Part Examined                  CS: 'CHEST'
(0018, 5101) View Position                       CS: 'AP'
(0020, 000d) Study Instance UID                  UI: 1.2.276.0.7230010.3.1.2.8323329.1556.1517874291.545551
(0020, 000e) Series Instance UID                 UI: 1.2.276.0.7230010.3.1.3.8323329.1556.1517874291.545550
(0020, 0010) Study ID                            SH: ''
(0020, 0011) Series Number                       IS: '1'
(0020, 0013) Instance Number                     IS: '1'
(0020, 0020) Patient Orientation                 CS: ''
(0028, 0002) Samples per Pixel                   US: 1
(0028, 0004) Photometric Interpretation          CS: 'MONOCHROME2'
(0028, 0010) Rows                                US: 1024
(0028, 0011) Columns                             US: 1024
(0028, 0030) Pixel Spacing                       DS: ['0.139', '0.139']
(0028, 0100) Bits Allocated                      US: 8
(0028, 0101) Bits Stored                         US: 8
(0028, 0102) High Bit                            US: 7
(0028, 0103) Pixel Representation                US: 0
(0028, 2110) Lossy Image Compression             CS: '01'
(0028, 2114) Lossy Image Compression Method      CS: 'ISO_10918_1'
(7fe0, 0010) Pixel Data                          OB: Array of 143458 bytes
```

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
