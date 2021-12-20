from math import inf
import numpy as np
import schedule
import threading
import shortuuid

from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from auto_training.Data import Data
from sklearn.model_selection import train_test_split

class AutoTraining:
  label_training = 250 # Số dữ liệu huấn luyện mô hình
  length_id = 10
  
  def __init__(self, 
    model : Sequential, 
    file_saved : str, 
    splited_char : str = '|', 
    path_saved_weight : str = "final_result.hdf5",
    callback_get_label = None, 
    callback_write_data = None,
    callback_read_data = None,
    callback_convert_X = None,
    epochs : int = 20,
    batch_size : int = None,
    loss = "binary_crossentropy",
    optimizer="sgd",
    metrics=["accuracy"]
  ):
    if isinstance(file_saved, str) == False:
      raise ValueError("file_saved must be str")
    self.model = model
    self.file_saved = file_saved
    self.splited_char = splited_char
    self.callback_write_data = callback_write_data
    self.callback_get_label = callback_get_label
    self.callback_read_data = callback_read_data
    self.callback_convert_X = callback_convert_X
    self.path_saved_weight = path_saved_weight
    self.history = +inf
    self.checkpoint = ModelCheckpoint(self.path_saved_weight, save_best_only=True, save_weights_only=True)
    self.data = Data()
    self.loss = loss
    self.optimizer = optimizer
    self.metrics = metrics
    self.__counter_new_data = 0
    self.__istraining = False
    self.__mem_called_read_data = False
        
    # Khởi tạo sự kiện huấn luyện mô hình
    schedule.every().sunday.at("12:00:00").do(self.__training, args=(epochs, batch_size)).tag("DEFAULT_EVENT")
              
  def run(self, X : np.ndarray):
    y_predict = self.model.predict(X)
    y_labels = self.callback_get_label(y_predict)
    
    threading.Thread(name="Start writing data", target=self.__write_data, args=(y_labels, X, )).start()
    
    return y_predict, y_labels
  
  def read_data(self):
    '''
      Gọi hàm read_data
    '''
    self.__mem_called_read_data = True
    self.data.read_data(self.file_saved, self.splited_char, self.callback_read_data)
    
  def set_label_training(self, lb_tra : int):
    if isinstance(lb_tra, int) == False:
      raise ValueError("Must be Z")
    AutoTraining.label_training = lb_tra
  
  def set_length_id(self, length : int):
    if isinstance(length, int) == False:
      raise ValueError("length must be int")
    AutoTraining.length_id = length
  
  def call_training_data(self, epochs : int, batch_size = None):
    '''
      Huấn luyện mô hình
    '''
    self.__training(epochs, batch_size)
  
  def get_history_training(self):
    '''
      Lấy lịch sử huấn luyện mô hình
    '''
    return self.history
  
  def get_all_data(self):
    '''
      Lấy dữ liệu
    '''
    return self.data.raw_data
  
  def __training(self, 
    epochs : int, 
    batch_size = None
  ):
    optimizer = self.optimizer
    loss = self.loss
    metrics = self.metrics
    self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    if self.__counter_new_data < AutoTraining.label_training:
      return
    if self.__istraining:
      return
    
    self.__counter_new_data = 0
    
    self.data.X = None # Set giá trị giải phóng ô nhớ
    self.data.y = None # Set giá trị giải phóng ô nhớ
    
    X, y = self.data.read_data(self.file_saved, self.splited_char, self.callback_read_data)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    self.__istraining = True
    self.history = self.model.fit(X_train, y_train, 
      validation_data=(X_val, y_val), 
      epochs=epochs, 
      batch_size=batch_size, 
      use_multiprocessing=True,
      callbacks=[self.checkpoint]
    )
    self.__istraining = False
    
  def __write_data(self, y_labels, X):
    if self.__mem_called_read_data == False:
      raise ValueError("You must call read_data before.")
    for i in range(len(X)):
      point_X = X[i]
      y = y_labels[i]
      
      id_ = shortuuid.ShortUUID().random(length=AutoTraining.length_id)
      temp = {}
      temp[id_] = {
        "X" : [str(i) for i in list(point_X)],
        "y" : y,
        "checked" : False
      }
      
      self.__counter_new_data += 1
      self.data.write_data(temp, self.file_saved, self.callback_write_data)
      
  def update_label(self, id_ : str, label : np.ndarray):
    if isinstance(id_, str) == False:
      raise ValueError("id must be str")
    if isinstance(label, np.ndarray) == False:
      raise ValueError("label must be numpy array")
    self.data.update_label(id_, label)