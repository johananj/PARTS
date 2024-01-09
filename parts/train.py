# import numpy as np
# import pandas as pd
# import pyarrow.parquet as pq
import time
import os
# import matplotlib.pyplot as plt
# from scipy import signal
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint, CSVLogger, Callback
from keras.layers import Conv1D, LSTM, MaxPool1D, Dense, Flatten, Dropout


def compile_model(dpo, ks = [9, 5], ds = [64, 32], arch = '00', framesize=None, feature_dim=None):
  '''
  dpo: An instance of the DataPrep class.
  ks: Kernel sizes. 
  ds: Deep layer sizes
  framesize: Framesize used in dataprep.
  feature_dim: Number of features extracted.
  '''
  if framesize is None:
    framesize = dpo.framesize
  if feature_dim is None:
    feature_dim = dpo.feature_dim
  
  model = Sequential()
  if arch == '00':
    model.add(Conv1D(filters=24, kernel_size=ks[0], activation='relu', input_shape=[framesize, feature_dim]))
    model.add(Conv1D(filters=24, kernel_size=ks[0], activation='relu'))
    model.add(MaxPool1D(pool_size=2))
    model.add(Dropout(rate=0.25))

    model.add(Conv1D(filters=48, kernel_size=ks[1], activation='relu'))
    model.add(Conv1D(filters=48, kernel_size=ks[1], activation='relu'))
    model.add(MaxPool1D(pool_size=2))
    model.add(Dropout(rate=0.25))
    
    model.add(Flatten())
    model.add(Dense(ds[0], activation='relu'))
    model.add(Dropout(rate=0.25))
    model.add(Dense(ds[1], activation='relu'))
    model.add(Dropout(rate=0.25))
    model.add(Dense(2, activation='softmax'))
  
  elif arch == '10':
    model.add(LSTM(18, return_sequences=True))
    model.add(LSTM(18))
    model.add(Dense(2, activation='softmax'))


  model.compile(
    loss='categorical_crossentropy', 
    optimizer='adam', 
    metrics=['accuracy']
  )
  print('Model Compiled')
  return model

def model_training(dpo, checkpoint_dir, model=None, num_epoch=5, load_model_flag=False, model_checkpoint=None, save_frequency=None):
  '''
  Inputs:
    dpo: An instance of the DataPrep class.
    checkpoint_dir: The directory where the trained model has to stored. 
      Creates it, if it does not exist.
    model: Pass a sequential model as in the compile_model() function, 
      if not, the default model will be used. 
    num_epoch: Number of epochs. 
    load_model_flag: Load an existing model by providing it's checkpoint directory, 
      to be used as a starting point. Not the directory where all checkpoints are stored, but the specific checkpoint with a specific loss.
    model_checkpoint: must be present if load_model_flag is set to True.
  '''
  if model is None:
    model = compile_model(dpo)
  
  if save_frequency is None:
    save_frequency = int(dpo.batch_size/4)
  
  num_batches = dpo.batch_size*num_epoch
  if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
  
  checkpoint_callback = ModelCheckpoint(filepath=checkpoint_dir + '/ckpt-loss={loss:.2f}', 
                                        save_freq=save_frequency)
  logger_callback = CSVLogger(filename=checkpoint_dir+ '/log.csv', append=True, separator=';')
  
  class SleepCallback(Callback):
    def on_train_batch_end(self, batch, logs=None):
      time.sleep(3)

    def on_epoch_end(self, epoch, logs=None):
      time.sleep(180)
      
  my_callbacks = [checkpoint_callback, 
                  logger_callback, 
                  SleepCallback(),]
      
  if load_model_flag == True:
    model = load_model(model_checkpoint)
    print('Using Latest Checkpoint')

  model.fit(x = dpo.generate_data(num_batches), 
          steps_per_epoch = dpo.batch_size, 
          epochs = num_epoch, callbacks = my_callbacks)
  return True
