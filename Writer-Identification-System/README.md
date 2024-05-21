# Writer-Identification-System

Writer identification system (written in Python) for identifying the writer of a handwritten paragraph image.

acknowledgement: the model was trained using IAM Handwriting Database. 


The pipline of our system is that we take the image then pass it to preprocessing module then take the preprocessed image and pass it the feature extraction module to get the features by using local binary patterns then pass these features to k-Nearest Neighbors KNN  classifier to tell us who is the writer of the image.

We tested this code on 40 test cases and we got 90% accuracy.


Libraries needed:
  - import cv2
  - import numpy as np
  - import matplotlib.pyplot as plt
  - from scipy import ndimage
  - from scipy.ndimage import label
  - import os
  - import statistics
  - from sklearn.neighbors import KNeighborsClassifier
  - import time


How to run the code:
  - It is a python code in pattern.py file.
  - Edit the code with any code editor to change the path to the data folder.
  - Run using python.
  - results.txt ad time.txt will be generated at the data folder.


- Link of 40 test cases:  https://drive.google.com/file/d/1ay27biVIsuU9kb7ZstaLQDBzfqbS6Nbn/view?usp=sharing

reference:
U. Marti and H. Bunke. The IAM-database: An English Sentence Database for Off-line Handwriting Recognition. Int'l Journal on Document Analysis and Recognition, Volume 5, pages 39 - 46, 2002.
