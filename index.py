from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from auto_training import AutoTraining

def callback_get_label(y):
  '''
    Kết quả trả về là str
  '''
  return ['0' if i <= 0.5 else '1' for i in y]

model = Sequential()

model.add(Dense(4, activation="relu" ,input_shape=(4,)))
model.add(Dense(1, activation="sigmoid"))

autoTrain = AutoTraining(model, "haha.json", callback_get_label=callback_get_label)
autoTrain.set_label_training(1)
autoTrain.read_data()

counter = 0

while True:
  s = list(map(int, input().split()))
  print (autoTrain.run(np.array([s])))
  counter += 1
  if counter == 1:
    autoTrain.call_training_data(20)
    counter = 0
  