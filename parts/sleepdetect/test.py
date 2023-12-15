import numpy as np
# import pandas as pd
# import pyarrow.parquet as pq
# import time, random
# import os
# import matplotlib.pyplot as plt
from scipy import signal
from keras.models import load_model

def test_one(dpo, model_checkpoint=None, model=None, group_idx=None, data_df=None,denoise=False):
  """
  Given a group or a dataframe. Test it with the model assigned as a class variable self.model.
  """
  if model is None: 
    model = load_model(model_checkpoint)
    print('Model Loaded:' + model_checkpoint)

  if group_idx is not None:
    x_test, y_test = dpo.extract_features(dpo.group_to_df(group_idx))
  elif data_df is not None:
    x_test, y_test = dpo.extract_features(data_df)
  else:
    print("No data or group id provided.")

  # print(x_test.shape, y_test.shape)
  prediction = model.predict(x = x_test)
  # print(prediction)

  if denoise is True:
    _, pred_pulse = prediction_pulse(prediction)
  else:
    pred_pulse, _ = prediction_pulse(prediction)

  return prediction, y_test, pred_pulse

def test_all(dpo,model_checkpoint,denoise=False):
  pred_all = np.zeros([2,2])
  y_test_all = np.zeros([2,2])
  pred_pulse_all = np.zeros([2,1])
  
  model = load_model(model_checkpoint)
  print('Model Loaded:' + model_checkpoint)
  
  for i in dpo.test_idxs:
    pred, y_test,pred_pulse = test_one(dpo,model=model,group_idx=i,denoise=denoise)
    pred_all = np.concatenate((pred_all, pred),axis=0)
    y_test_all = np.concatenate((y_test_all, y_test),axis=0)
    pred_pulse_all = np.concatenate((pred_pulse_all, pred_pulse),axis=0)
    print(pred_all.shape, y_test_all.shape, pred_pulse_all.shape)

  pred_all = np.delete(pred_all, [0,1], axis=0)
  y_test_all = np.delete(y_test_all, [0,1], axis=0)
  pred_pulse_all = np.delete(y_test_all, [0,1], axis=0)
  return pred_all, y_test_all, pred_pulse_all


def prediction_pulse(prediction, threshold = 0.25, denoise_kernel_size = 31):
  """
  Convert the activations from the neural network to a pulse-like waveform
  """
  pred_pulse = prediction[:,0].copy()
  pred_pulse[pred_pulse > threshold] = 1
  pred_pulse[pred_pulse <= threshold] = 0
  pred_pulse_d = signal.medfilt(pred_pulse, kernel_size = denoise_kernel_size)
  pred_pulse = np.reshape(pred_pulse, (pred_pulse.shape[0],1))
  pred_pulse_d = np.reshape(pred_pulse_d, (pred_pulse_d.shape[0],1))
  return pred_pulse, pred_pulse_d