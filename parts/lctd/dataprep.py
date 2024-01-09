import numpy as np
import math, os, time, statistics, random
import opensmile
from keras.utils import to_categorical

class DataPrep:
  def __init__(self, feature=None, train_list=None, test_list=None, framesize=None, mini_batchsize=50):
    '''
    train_list: List. list of train files.
    test_list: List. list of test files.
    feature: String. 'fs01' or 'fs02'.
    framesize: Integer. Maximum number of feature frames to consider as one example \
      This is with respect to the NN, not the time-domain signal. \
      Each sample here is by itself derived from a frame of time-domain signal. 
    mini_batchsize: Integer. Number of files to consider for each batch.
    '''
    if feature is None:
      self.feature_index = 'fs01' 
      print("Feature set to: fs01")
    else:
      self.feature_index = feature
    
    self.train_list = np.array(train_list)
    self.test_list = np.array(test_list)
    if self.train_list is None and self.test_list is None:
      print("No train or test files provided.")

    self.smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    )
    # feature sets
    self.featureset = {
      "fs01" : ['F0final_sma', 'voicingFinalUnclipped_sma', 'pcm_RMSenergy_sma', \
      'pcm_zcr_sma', 'logHNR_sma', 'jitterDDP_sma', 'jitterLocal_sma', \
      'shimmerLocal_sma', 'pcm_fftMag_spectralFlux_sma', 'pcm_fftMag_psySharpness_sma'],
      
      "fs02" : ['mfcc_sma[1]',	'mfcc_sma[2]',	'mfcc_sma[3]',	\
      'mfcc_sma[4]',	'mfcc_sma[5]',	'mfcc_sma[6]',	\
      'mfcc_sma[7]',	'mfcc_sma[8]',	'mfcc_sma[9]',	\
      'mfcc_sma[10]',	'mfcc_sma[11]',	'mfcc_sma[12]',	\
      'mfcc_sma[13]',	'mfcc_sma[14]']
    }
    
    self.feature_dim = len(self.featureset[self.feature_index])
    if framesize is None: 
      self.framesize = 2000
    else:
      self.framesize = framesize 
    if self.train_list is not None:
      self.batch_size = len(self.train_list)
    self.mini_batchsize = mini_batchsize


  def extract_features(self, filepath, feature_index="fs01"):
    '''
    extract features from one single input file
    filepath: full path of the wav file.
    feature_index: fs01 or fs02
    '''
    if filepath is None: 
      print("provide filepath")
      return False
    else: 
      x_data_df = self.smile.process_file(filepath)
      # the number of rows/frames is obtained from a random (here first) series from the data frame.
      feature_matrix = np.zeros([np.shape(x_data_df[self.featureset[feature_index][0]].values)[0], self.feature_dim])
      # accumulate the required features into the feature matrix, based on the feature set.
      for i in range(self.feature_dim):
        feature_matrix[:,i] = x_data_df[self.featureset[feature_index][i]].values
    return feature_matrix

  def get_framelabels(self, filepath, num_frames):
    '''
    - get the target label of an example from filename.
    - size based on num_frames
    - num_frames: number of frames in x_data
    '''
    labels = np.zeros([num_frames,1])
    if filepath.find("TA0") > 0:
      labels = labels + 0
    else:
      labels = labels + 1
    return labels
  
  def get_label(self, filepath):
    if filepath.find("TA0") > 0:
      labels = 'CT'
    else:
      labels = 'LT'
    return labels

  def extract_features_batch(self, wavefile_list):
    '''
    - extract features for a batch of files in a list
    '''
    if np.shape(wavefile_list)[0] == 1:
      # Single file. For Testing. batch size should be 1
      for i in np.arange(np.shape(wavefile_list)[0]):
        wavefile_path = wavefile_list[i]
        # print(wavefile_path)
        S = self.extract_features(wavefile_path, self.feature_index)
        S = np.reshape(S, [1,np.shape(S)[0],np.shape(S)[1]])
        x_data = S
        labels = self.get_framelabels(wavefile_path, 1)
        labels = to_categorical(labels, 2)
        y_data = labels
        # print(np.shape(S), np.shape(labels))
        # print(wavefile_path, labels, np.shape(x_data), np.shape(y_data))
        # time.sleep(1)
        # print('==')
    else: 
      # Multiple files. For Training.
      x_data = np.zeros([2,self.framesize,self.feature_dim])
      y_data = np.zeros([2,1])

      for wavefile_path in wavefile_list:
        # print(wavefile_path)
        S = self.extract_features(wavefile_path, self.feature_index)
        # print(np.shape(S))
        S = self.reshape_x(S)
        # print(np.shape(S))
        # time.sleep(1)
        x_data = np.concatenate((x_data, S), axis=0)
        
        labels = self.get_framelabels(wavefile_path, np.shape(S)[0])
        y_data = np.concatenate((y_data, labels), axis=0)
        # print(wavefile_path, labels, np.shape(x_data), np.shape(y_data))
        # time.sleep(0.1)
        # print('==')

      x_data = np.delete(x_data, [0,1], axis=0)
      y_data = np.delete(y_data, [0,1], axis=0)

      # convert integers to dummy variables (i.e. one hot encoded)
      y_data = to_categorical(y_data, 2)

    print(np.shape(x_data))
    print(np.shape(y_data))
    return x_data, y_data  
  
  def reshape_x(self, feature_frames):
    '''
    - Reshape feature frames, based on framesize. 
    - 2D matrices to 3D matrices.
    feature_frames: feature frames from extract_features_batch usually.
    '''
    if self.framesize > feature_frames.shape[0]:
      # zero padding, base case
      zeropad_rows = self.framesize - feature_frames.shape[0]
      feature_frames = np.concatenate((feature_frames, np.zeros([zeropad_rows, self.feature_dim])), axis=0)
    else: 
      # truncating
      feature_frames = feature_frames[:self.framesize,:]
    feature_frames = feature_frames.reshape(1,self.framesize,self.feature_dim)
    return feature_frames

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

