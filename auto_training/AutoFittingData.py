from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
import cv2
import json
import numpy as np
import os

class AutoFittingData:
  def __init__(self, 
    model : Sequential, 
    saved_label : str = "final_label.json", 
    file_saving = "final_model.h5", 
    checkpoint_file_name = "final_heights.hdf5",
    epochs : int = 100,
    batch_size : int = None,
    callback_convert_data = None
  ):
    if isinstance(saved_label, str) == False:
      raise ValueError("saved_label must be str")
    if self.__checked_format_file(saved_label) == False:
      raise ValueError("Only access json file")
    self.model = model
    self.folder = ""
    self.labels = {}
    self.saved_label = saved_label
    self.file_saving = file_saving
    self.checkpoint_file_name = checkpoint_file_name
    self.epochs = epochs
    self.batch_size = batch_size
    self.callback_convert_data = callback_convert_data
    self.width = 124
    self.height = 124
    self.model_checkpoint = ModelCheckpoint(self.file_saving, save_best_only=True, save_weights_only=True)
    self.X = []
    self.y = []
        
  def loading_folder(self, folder : str, block_training : bool = False):
    if isinstance(folder, str) == False:
      raise ValueError("folder's param must be str")
    self.folder = folder
    self.list_data = os.listdir(folder)
    self.__build_label()
    self.__loaddata()
    if block_training == False:
      self.training()
    
  def __build_label(self):
    c = 0
    for label in self.list_data:
      self.labels[label] = c
      c += 1
    with open(self.saved_label, "wb", encoding="utf-8") as fout:
      json.dump(self.labels, fout, ensure_ascii=False)
      
  def __checked_format_file(str, file : str):
    return file.split('.')[1] == "json"
  
  def __loaddata(self):
    for label in self.list_data:
      print ('Handling folder name = {}'.format(label))
      current_path = os.path.join(self.folder, label)
      files = os.listdir(current_path)
      for file in files:
        print ("RUNNING FILE = {}".format(file))
        X, y = self.callback_convert_data(os.path.join(current_path, file), label ,len(list(self.labels.keys())))
        self.X.append(X)
        self.y.append(y)
        print ("HANDLING SUCCESSFULLY!")
    self.X = np.array(X)
    self.y = np.array(y)
  
  def build_get_data(self, function_):
    self.callback_convert_data = function_
    
  def get_data_default(self, filename : str, labels : int, folder_name : str, types="image"):
    '''
      Chạy với ảnh
    '''
    if isinstance(filename, str) == False:
      raise ValueError("filename must be str")
    if isinstance(labels, int) == False:
      raise ValueError("labels must be int")
    if isinstance(folder_name, str) == False:
      raise ValueError("folder_name must be str") 
    if types == "images":
      image_read = cv2.imread(filename)
      image_read = cv2.resize(image_read, (self.width, self.height))
      image_read = np.array(image_read) / 255
      label_ = self.labels[folder_name]
      
      if labels == 1:
        return image_read, [label_]
      
      y_ = np.zeros((labels))
      y_[label_] = 1
      return image_read, y_
    
    raise ValueError("Không thèm lấy")
  
  def set_width_height(self, X : tuple):
    self.width, self.height = X
    
  def training(self):
    self.model.fit(self.X, self.y, 
      epochs=self.epochs, 
      batch_size=self.batch_size, 
      use_multiprocessing=True,
      callbacks=[self.model_checkpoint]
    )
    self.model.save(self.file_saving)