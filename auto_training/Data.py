import schedule
import numpy as np
import json

class Data:
  accepted_file = ["json"]
  
  def __init__(self):
    self.X = []
    self.y = []
    self.raw_data = {}

    schedule.every().sunday.at("10:00:00").do(self.__write_default).tag("WRITE_DEFAULT")
    
  def __load_file(self, file : str, splited_char : str = '|'):
    if isinstance(file, str) == False:
      raise ValueError("File's param must be str")
    if isinstance(splited_char, str) == False:
      raise ValueError("splited_char's param must be str")
    
    self.file = file
    self.splited_char = splited_char
  
  def __prepare(self):
    format_ = self.file.split('.')[1]
    accepted = False
    
    for f in self.accepted_file:
      if f == format_:
        accepted = True
        break
      
    if accepted == False:
      raise ValueError("Bạn truyền vào loại file không được chấp nhận")
    
    return format_
      
  def __read_data(self, format_ : str):
    if isinstance(format_, str) == False:
      raise ValueError("format_'s param must be str")
    
    X = []
    y = []
    
    with open(self.file, "r") as fin:
      if format_ == self.accepted_file[0]:
        raw_data = json.load(fin)
        self.raw_data = raw_data
        # Load lại dữ liệu
        tempoary = raw_data["data"]
        ids = list(tempoary.keys())
        for id_ in ids:
          if tempoary[id_]["checked"]: # Dữ liệu đã chứng thực
            X.append(list(map(int, tempoary[id_]["X"])))
            y.append(list(map(int, tempoary[id_]["y"])))
                    
      elif format_ == self.accepted_file[1]:
        while True:
          point_data = fin.readline()
          
          if len(point_data) == 0:
            break
          
          raw_x, raw_y = point_data.split(self.splited_char)
          raw_x = map(float, raw_x.split(','))
          raw_y = map(int, raw_y.split(','))
          
          X.append(raw_x)
          y.append(raw_y)
    
    self.X = X
    self.y = y
  
  def read_data(self, file : str = '', splited_char : str = '|', read_data_function = None):
    if read_data_function == None:
      self.__load_file(file, splited_char)
      
      format_ = self.__prepare()
      self.__read_data(format_)
    else:
      self.X, self.y, self.raw_data = read_data_function()
    return np.array(self.X), np.array(self.y)
  
  def write_data(self, data : dict, file : str = '', write_data_function = None):
    if isinstance(data, dict) == False:
      raise ValueError("data must be dict")
    if write_data_function == None:
      self.file = file
      format_ = self.__prepare()
      with open(file, "w", encoding="utf-8") as fout:
        if self.raw_data.get("data") == None:
          self.raw_data["data"] = {}
        self.raw_data["data"][list(data.keys())[0]] = data[list(data.keys())[0]]
        json.dump(self.raw_data, fout, ensure_ascii=False)
    else:
      write_data_function(data)
  
  def __write_default(self):
    '''
      Ghi dữ liệu vô file định kì
    '''
    with open(self.file, "wb") as fout:
      json.dump(self.raw_data, fout)
  
  def update_label(self, id_ : str, label : int):
    self.raw_data["data"][id_]["y"] = label
    
  def data_is_passport(self, id_):
    if isinstance(id_, str) == False:
      raise ValueError("id_'s param must be str")
    
    self.raw_data["data"][id_].checked = True