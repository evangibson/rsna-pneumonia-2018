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
dcm_file = 'path/to/dicom/file.dcm'
dcm_data = pydicom.read_file(dcm_file)
print(dcm_data)
```
```
(0008, 0005) Specific Character Set              CS: 'ISO_IR 100'
(0008, 0016) SOP Class UID                       UI: Secondary Capture Image Storage,
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

<p align="center">
  <h4> Frequency Chart of Detailed Class (Colored by Binary Class) </h4>
  <img width="240" height="100" src="Images/detailed_freq_chart.png?raw=true "Detailed Frequency Chart">
  <h4> Frequency Chart of Binary Class (Colored by Binary Class) </h4>
  <img width="240" height="100" src="Images/binary_freq_chart.png?raw=true "Binary Frequency Chart">
  <h4> Frequency Chart of Sex (Colored by Binary Class) </h4>
  <img width="240" height="100" src="Images/sex_freq_chart.png?raw=true "Sex Frequency Chart">
</p>
  
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

Data is transmitted and understood by computers as a series of ones and zeros - otherwise known as *binary* - and this doesn't stop with images:

To store an image on a computer, the image is broken into elements called pixels (short for picture element) - which represent one color each.  This is where binary comes into play.  In order for the computer to store images, each pixel is represented by a binary value.  This representation of colors is called the *bit-plane*.  

> The bit-plane essentially defines the number of available colors in the image - each bit doubles the amount of colors the image can have.  For example, 1-bit would give us two colors, 2-bit would give us 4, 3-bit would give us 8, etc...

##### Computer vision  

Like teaching a newborn what something is in the world, we also have to teach our computers what an object is using the same approach - through *examples*.  

After an image is stored on a computer, it still has no idea what the image *actually* is - outside of the binary array that it is stored as.  This is where computer vision comes into play.

The next step in having your computer identify what it's looking at is by training it to do so.  This is done by feeding your computer thousands (or more) of labeled or pre-identified images.  These images are used to train deep-learning models, which essentially automate the process of learning what series of bits correlate to which images.

Today there are a plethora of computer vision types that are used in different ways, here are a few:

> **Image segmentation**: partitions an image into multiple regions or pieces that will be examined separately.  
>
> **Object detection**: Identifies a specific object in an image.  Advanced object detection can recognize many objects in a single image.  These models use an X,Y coordinate to create a *bounding box* and identify everything inside the box.  
> 
> **Facial recognition**:  An advanced type of object detection that recognizes a human face in an image, while also identifying specific individuals.  
> 
> **Edge detection**: Used to identify the outside edge of an object or landscape to better identify what is in the image.  
> 
> **Pattern detection**: A process of recognizing repeated shapes, colors, and other visual indicators in images.  
> 
> **Image classification**: Groups images into different categories  
> 
> **Feature matching**: A type of pattern detection that matches similarities in images to help classify them.  

For our purposes, we will be mostly concerned with the different algorithms designed for object detection - using a bounding box to identify which areas of pneumonia-riddled lungs are unhealthy.  

---
#### Key Metrics from the Bounding Box Detector *(Bounding Box Predictor (CNN).ipynb)*
🔧 _Image incoming_ 🔧

#### Key Metrics from a Binary Classifier *(Binary Classifier 2.ipynb)*

🔧 _Image incoming_ 🔧

It's pretty clear that, for now, the Bounding Box Detector *does not* need any help from my current classifiers. 

___
## Acknowledgments and References
