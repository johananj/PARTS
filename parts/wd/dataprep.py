import numpy as np
import pandas as pd
import math, os, time, statistics, random, re
import librosa
from scipy import fft
from keras.utils import to_categorical


class DataPrep:
  def __init__(self, feature, train_list=None, test_list=None, 
               bin_start=0, bin_end=128, mini_batchsize=50):
    '''
    train_list: List. list of train files.
    test_list: List. list of test files.
    feature: String. "stft|mfcc|lfbe".
    framesize: Integer. Number of bins considered.
    mini_batchsize: Integer. Number of files to consider for each batch.
    '''
    if train_list is None and test_list is None:
      print("No train or test files provided.")
    elif train_list is not None:
      self.train_list = np.array(train_list)
      self.batch_size = len(self.train_list)
    elif test_list is not None:
      self.test_list = np.array(test_list)
    
    self.feature=feature
    self.feature_dim = 1
    self.mini_batchsize = mini_batchsize
    # sampling rate. feature spec.
    self.resample_rate = 44100
    self.fft_size = 1024
    # bin start and bin end. quarter spec.
    self.bin_start = bin_start
    self.bin_end = bin_end
    if self.feature == "stft":
      self.framesize = self.bin_end - self.bin_start
    else:
      self.framesize = 64
    self.snr = 0

  def get_framelabels(self, filepath, num_frames):
    '''
    get the label of a file.
    distribute it across every frame.
    '''
    labels = np.zeros([num_frames,1])
    if filepath.find("n.WAV") > 0:
        labels = labels + 0
    elif filepath.find("w.WAV") > 0:
        labels = labels + 1
    else:
        print('unknown')
    return labels

  def add_noise(self, xc, xn):
    len_xc = np.shape(xc)[0]
    xc_power = np.mean(xc ** 2)/len_xc
    # xc_power_db = 10 * np.log10(xc_power)
    xn = xn[0:len_xc]
    xn_power = np.mean(xn ** 2)/np.shape(xn)[0]
    # xn_power_db = 10 * np.log10(xn_power)
    sf= np.sqrt(xc_power/xn_power/(np.power(10,(self.snr/10))));
    xn = xn * sf; 
    x = xc + xn
    # print(10*np.log10(np.dot(xc,xc)/np.dot(xn,xn)))
    return x

  def white_noise(self, len_xc, mean_noise = 0, var_noise = 1, ):
    x = np.random.normal(mean_noise, var_noise, len_xc)
    return x
  
  def extract_stft(self, filepath):
    '''
    Extract the stft for a single file. 

    Input
    filepath: wav file path.
    
    Output 
    S: numpy array of features.
    '''
    ## load wavefiles
    # print(filepath)
    x, fs = librosa.load(filepath, sr=self.resample_rate)
    if self.snr != 0:
      x = self.add_noise(x, self.white_noise(np.shape(x)[0], 0, 1))
    ## compute the stft
    S = np.abs(librosa.stft(x, n_fft=self.fft_size, hop_length=int(self.fft_size/8)))
    S = librosa.amplitude_to_db(S)
    ## select bins
    S = S[self.bin_start:self.bin_end,:]
    return S

  def extract_mfcc(self, filepath, n_mel=64):
    '''
    Extract the mfcc for a single file.

    Inputs 
    filepath: wav file path.
    n_mel: number of mel filters 

    Output 
    S: numpy array of features
    '''
    ## load wavefiles
    x, fs = librosa.load(filepath, sr=self.resample_rate)
    if self.snr != 0:
      x = self.add_noise(x, self.white_noise(np.shape(x)[0], 0, 1))
    ## compute mfcc
    # Sm = librosa.feature.mfcc(y=x, sr=fs, n_mfcc=n_mel)
    S = np.abs(librosa.stft(x, n_fft = self.fft_size, hop_length=int(self.fft_size/8)))
    S = librosa.amplitude_to_db(S)
    S = librosa.feature.melspectrogram(S=S, sr=fs)
    S = librosa.feature.mfcc(S=S, n_mfcc = n_mel)
    return S

  def extract_lfbe(self, filepath, n_mel=64):
    '''
    Extract the lfbe for a single file.
    
    Inputs 
    filepath: wav file path.
    n_mel: number of mel filters 
    
    Output 
    S: numpy array of features
    '''
    ## load wavefiles
    x, fs = librosa.load(filepath, sr=self.resample_rate)
    if self.snr != 0:
      x = self.add_noise(x, self.white_noise(np.shape(x)[0], 0, 1))
    ## compute mfcc
    # Sm = librosa.feature.mfcc(y=x, sr=fs, n_mfcc=n_mel)
    S = np.abs(librosa.stft(x, n_fft = self.fft_size, hop_length=int(self.fft_size/8)))
    S = librosa.amplitude_to_db(S)
    S = librosa.feature.melspectrogram(S=S, sr=fs, n_mels = n_mel)
    # S = np.log10(S)
    # S = librosa.feature.mfcc(S=S, n_mfcc = n_mel)
    return S

  def extract_features_batch(self, wavefile_list, feature=None): 
    '''
    Extract a batch of features. 

    Input
    wavefile_list: list of wave file paths.

    Output 
    x_data: numpy array of features 
    y_data: corresponding labels
    '''     
    x_data = np.zeros([2,self.framesize])
    y_data = np.zeros([2,1])

    for wavefile_path in wavefile_list:
      # print(wavefile_path)
      if self.feature == "stft":
        S = self.extract_stft(wavefile_path)
      elif self.feature == "mfcc":
        S = self.extract_mfcc(wavefile_path, n_mel = 64)
      elif self.feature == "lfbe":
        S = self.extract_lfbe(wavefile_path, n_mel = 64)
        
      labels = self.get_framelabels(wavefile_path, np.shape(S)[1])
      x_data = np.vstack((x_data,np.transpose(S)))
      y_data = np.vstack((y_data, labels))
      # time.sleep(0.1)
    
    x_data = np.delete(x_data, [0,1], axis=0)
    y_data = np.delete(y_data, [0,1], axis=0)
    
    # convert integers to dummy variables (i.e. one hot encoded)
    y_data = to_categorical(y_data, 2)
    x_data = x_data.reshape(x_data.shape[0], x_data.shape[1], 1)
    # print(np.shape(x_data))
    # print(np.shape(y_data))
    return x_data, y_data

  def generate_data(self, num_batches):
    for i in range(num_batches):
      time.sleep(1)
      wavefiles_select = np.random.randint(self.batch_size , size=(self.mini_batchsize))
      print(wavefiles_select)
      print(type(wavefiles_select))
      wavefile_list = self.train_list[wavefiles_select]
      print("\nBatch " + str(i) + " out of " + str(num_batches))
      x_train, y_train = self.extract_features_batch(wavefile_list)
      yield x_train, y_train

  
