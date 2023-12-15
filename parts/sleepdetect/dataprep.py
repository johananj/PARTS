import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import time, random, os
import matplotlib.pyplot as plt
from scipy import signal
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.layers import Conv1D, MaxPool1D, Dense, Flatten, Dropout

class DataPrep:
  def __init__(self, datafile_path=None, label_file_path=None):
    # initialization
    # self.datafile_path = './train_series.parquet'
    # self.labelfile_path = "./train_events.csv"
    self.datafile_path = datafile_path
    self.labelfile_path = label_file_path
    if self.datafile_path is not None:
      self.datafile = pq.ParquetFile(self.datafile_path)
      print(self.datafile.metadata)
    else:
      print("No data file provided.")
    if self.labelfile_path is not None:
      self.labels_df = pd.read_csv(self.labelfile_path).dropna()
    else:
      print("No label file provided.")
    # feature extraction
    self.framesize = 240
    self.frameshift_factor = 0.25
    self.feature_dim = 2
    self.anglez_max = 90
    self.anglez_min = -90
    self.enmo_max = 0.5
    self.enmo_min = 0
    self.mixed_labels=2
    self.train_idxs = np.concatenate([np.arange(0,122), 
                                      np.arange(122,244), 
                                      np.arange(244,366)])
    self.test_idxs = np.concatenate([np.arange(366,488)])
    self.batch_size = self.train_idxs.shape[0]

  def get_label(self, series_id, start_step, end_step):
    """
    Outputs a label based on the self.series_id, start and end steps.
    Algorithm:
    if the start and end indexes are both 
      before and closest to an onset, then it is an awake cycle. 
      before and closest to a wakeup, then it is a sleep cycle.
    if the start and end indexes are both 
      before and closest to different onset and wakeup, then its a mixed cycle
    if a mixed cycle has proportionately more awake, then it's an awake cycle,
      otherwise it is a sleep cycle

    Awake = 1
    Sleep = 0
    """
    labels_df_filt = self.labels_df[self.labels_df['series_id']==series_id].copy()
    # print(labels_df_filt)
    
    # find the nearest start by subtracting current index with all index
    labels_df_filt['nearest_start'] = labels_df_filt['step'] - start_step
    labels_df_filt['nearest_end'] = labels_df_filt['step'] - end_step
    # print(labels_df_filt)
    try:
      # chose the ones closest to the start and end
      start_idx = labels_df_filt['nearest_start'][labels_df_filt['nearest_start']>0].idxmin()
      end_idx = labels_df_filt['nearest_end'][labels_df_filt['nearest_end']>0].idxmin()
      if start_idx == end_idx:
        if labels_df_filt['event'].loc[start_idx] == "onset":
            label=1 #awake
        elif labels_df_filt['event'].loc[start_idx] == "wakeup":
            label=0 #sleep
      else:
        if self.mixed_labels > 1:
          event_ratio = np.abs((labels_df_filt['step'].loc[start_idx]-start_step)/ \
                  (labels_df_filt['step'].loc[start_idx]-end_step))
          if labels_df_filt['event'].loc[start_idx] == "onset":
            if event_ratio >= 1:
                label=1 #awake
            else: 
                label=0 #sleep
          elif labels_df_filt['event'].loc[start_idx] == "wakeup":
            if event_ratio >= 1:
                label=0 #sleep
            else: 
                label=1 #awake
        else:
          return False
    except:
        if self.mixed_labels > 0:
          start_idx = labels_df_filt['nearest_start'].idxmin()
          if labels_df_filt['event'].loc[start_idx] == "onset":
            label=1 #awake
          elif labels_df_filt['event'].loc[start_idx] == "wakeup":
            label=0 #sleep
        else:
          return False     
    return np.zeros([1,1]) + label

  def group_to_df(self, row_group):
    """Get row group and conver to dataframe"""
    datafile_rg = self.datafile.read_row_group(row_group)
    datafile_rg_df = datafile_rg.to_pandas()
    return datafile_rg_df

  def extract_features(self, data_df, normalization=False):
    """"Get dataframe, output x and y data."""
    frameshift = int(np.round(self.framesize * self.frameshift_factor))
    if normalization == True:
      data_df['anglez'] = (data_df['anglez'] - self.anglez_min)/(self.anglez_max - self.anglez_min)
      data_df['enmo'] = (data_df['enmo'] - self.enmo_min)/(0.5 - self.enmo_min)
    # print(data_df)
    
    x_data = np.zeros([2,self.framesize,self.feature_dim])
    y_data = np.zeros([2,1])

    no_ydata = 0
    
    for i in np.arange(0,data_df.shape[0]-self.framesize,frameshift):
      start_idx = i
      end_idx = i+self.framesize-1
      x_frame = data_df.loc[start_idx:end_idx]
      # print(x_frame)
  
      # verify if it contains only one series_id. else skip
      if x_frame['series_id'].unique().shape[0] > 1:
          # print("more than one")
          continue
      else:
          series_id = x_frame['series_id'].unique()[0]    
      # print(i, series_id)

      # getting x_data, 3 - anglez, 4 - enmo
      x_data_frame = x_frame.iloc[:,3:].values
      x_data_frame = np.reshape(x_data_frame, 
                                (1, x_data_frame.shape[0], 
                                  x_data_frame.shape[1]))

      # getting y_data
      start_step = x_frame.iloc[0,:]['step']
      end_step = x_frame.iloc[-1,:]['step']

      # if there is no y_data, then skip to the next frame
      try:
          label = self.get_label(series_id, start_step, end_step)
          # print(start_step, end_step, label)
      except:
          no_ydata = no_ydata + 1
          continue

      x_data = np.vstack((x_data, x_data_frame))
      y_data = np.vstack((y_data, label))
      # time.sleep(5)
        
    x_data = np.delete(x_data, [0,1], axis=0)
    y_data_raw = np.delete(y_data, [0,1], axis=0)

    if no_ydata > 0:
      print("No y_data frames: ", no_ydata)
    y_data = to_categorical(y_data_raw, 2)
    print(x_data.shape, y_data.shape)
    
    return x_data, y_data
  
  def generate_data(self, num_batches):
    for i in range(num_batches):
      time.sleep(2)
      print("\nBatch " + str(i) + " out of " + str(num_batches))
      group_idx = self.train_idxs[random.randint (0, self.train_idxs.shape[0]-1)]
      # print(group_idx)
      x_train, y_train = self.extract_features(self.group_to_df(group_idx))
      while x_train.shape[0] == 0:
        group_idx = self.train_idxs[random.randint (0, self.train_idxs.shape[0]-1)]
        x_train, y_train = self.extract_features(self.group_to_df(group_idx))
      yield x_train, y_train