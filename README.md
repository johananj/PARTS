![Static Badge](https://img.shields.io/badge/under-construction-orange)

# PARTS - PAttern Recognition for Time Series data. 

## What and Why
- PARTS is a pattern recognition toolkit written in python, to quickly evaluate pattern recognition accuracies.
- It is based primarily on the 1DCNN. Something I have immense faith on, when it comes to time-series pattern classification.
- To carry out the evaluation, a series of steps are required: data preparation, training, testing.
- In cases involving feature extraction (instead of raw data), it is carried out live during the training phase, for each batch.
- An example use case can be found in example.ipynb

## Citation
If you use this code for your work, please consider citing the corresponding paper(s): 
1. [Quartered Spectral Envelope and 1D-CNN-Based Classification of Normally Phonated and Whispered Speech](https://link.springer.com/article/10.1007/s00034-022-02263-5)
2. [Literary and Colloquial Tamil Dialect Identification](https://link.springer.com/article/10.1007/s00034-022-01971-2)

## TO-DO
- [x] Example python notebook. 
- [x] Include a basic LSTM architecture.
- [ ] Data preparation classes for:
    - [ ] Whisper-normal speech classification using W-TIMIT.
    - [x] Literary-colloquial Tamil speech classification using the Microsoft Dataset.
    - [x] Sleep detection.
