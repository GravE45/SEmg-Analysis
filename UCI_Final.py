import os
import pandas as pd
import numpy as np
from keras.utils import to_categorical
import keras
from scipy import stats
from keras import Sequential,layers,Input,Model,optimizers
from sklearn.preprocessing import StandardScaler
from keras.layers import Conv1D, TimeDistributed, LSTM, Dense, Flatten, LeakyReLU, MaxPooling1D, Bidirectional, Dropout
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import load_model
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from keras.regularizers import l2,l1
from keras.layers import BatchNormalization, Activation
import tensorflow as tf

def step_decay(epoch):
  initial_lrate = 1e-4
  drop = 0.1
  epochs_drop = 70.0
  lrate = initial_lrate * tf.math.pow(drop,  
          tf.math.floor((1+epoch)/epochs_drop))
  return lrate

def create_segments_and_labels(df, time_steps,step,n_features):
    segments = []
    labels = []
    for i in range(0, len(df) - time_steps, step):
      for j in range(0, n_features):
        segments.append(df[:,j][i: i + time_steps])
      label = stats.mode(df[:,-1][i: i + time_steps])[0]
      labels.append(label)
    reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, time_steps, n_features)
    labels = np.asarray(labels)
    return reshaped_segments, labels

time_step = 1500
step = 1500
num_features = 8

def list_subfolders(folder_path):
    subfolders = []
    
    for root, dirs, files in os.walk(folder_path):
        # 'dirs' is a list of subdirectories in the current 'root' folder
        subfolders.extend(dirs)
        
        # If you only want to look in the immediate directory (not sub-subfolders), you can break here.
        # break

    return subfolders

values_to_remove = [0, 7]

def remove_rows_with_values(array, column_index, values_to_remove):
    mask = ~np.isin(array[:, column_index], values_to_remove)
    filtered_array = array[mask]
    return filtered_array

def remove_consecutive_duplicate_rows(dataframe):
    consecutive_duplicate_mask = dataframe.duplicated().shift(fill_value=False)
    return dataframe.loc[~consecutive_duplicate_mask]

def split_to_train_and_test(df1_filtered,df2_filtered):
    for i in range(0,df1_filtered.shape[0]-1):
        a = df1_filtered[i,-1]
        b = df1_filtered[i+1,-1] 

        if a == 6 and b ==1:
            border1 = i+1

    rep1 = df1_filtered[:border1,:]
    rep2 = df1_filtered[border1:,:]
    
    for i in range(0,df2_filtered.shape[0]-1):
        c = df2_filtered[i,-1]
        d = df2_filtered[i+1,-1]
        
        if c == 6 and d ==1:
            border2 = i+1
    
    rep3 = df2_filtered[:border2,:]
    rep4 = df2_filtered[border2:,:]

    return np.vstack((rep1,rep3,rep4)),rep2

data = []
train = []
test = []
train_final=  np.empty((0,10))
test_final=  np.empty((0,10))
train_filtered=  np.empty((0,10))
test_filtered=  np.empty((0,10))

dir = r'C:\Users\pc01\Desktop\Sid\Work\EMG_Data'
subfolders = list_subfolders(dir)

# print(subfolders)

for i in subfolders:
    dir_list = os.listdir(dir+'\\'+i)
    df1 = pd.read_csv(dir+'\\'+i+'\\'+dir_list[0], delimiter='\t').values
    df2 = pd.read_csv(dir+'\\'+i+'\\'+dir_list[1], delimiter="\t").values

    df1_filtered = remove_rows_with_values(df1,-1,values_to_remove)
    df2_filtered = remove_rows_with_values(df2,-1,values_to_remove)

    train, test = split_to_train_and_test(df1_filtered,df2_filtered)
    train_final = np.vstack((train_final,train))
    test_final = np.vstack((test_final,test))

tain_final = train_final[~np.any(np.isnan(train_final), axis=1)]
test_final = test_final[~np.any(np.isnan(test_final), axis=1)]

# for value, count in zip(unique_train, count_train):
#     print(f"Value {value} appears {count} times in train.")

# for value, count in zip(unique_test, count_test):
#     print(f"Value {value} appears {count} times in test.")


#selecting the required number of rows
for i in range(1,7):
    train_selection = train_final[train_final[:,-1] == i]
    test_selection = test_final[test_final[:,-1] == i]

    train_filtered = np.vstack((train_filtered,train_selection[:162000,:]))
    test_filtered = np.vstack((test_filtered,test_selection[:54000,:]))

# print(train_filtered.shape)
# print(test_filtered.shape)

#Dropping the time column
train_filtered = train_filtered[:,1:]
test_filtered = test_filtered[:,1:]

# print(train_filtered[:10,:])

#Scaling the data
sc = StandardScaler()
train_filtered[:,:-1] = sc.fit_transform(train_filtered[:,:-1])
test_filtered[:,:-1] = sc.transform(test_filtered[:,:-1])

x_train, y_train = create_segments_and_labels(train_filtered,time_step,step,num_features)
x_test, y_test = create_segments_and_labels(test_filtered,time_step,step,num_features)

y_train_encoded = y_train - 1
y_test_encoded = y_test - 1

num_classes = 6

y_train_onehot = to_categorical(y_train_encoded,num_classes)
y_test_onehot = to_categorical(y_test_encoded,num_classes)

# print(y_train_onehot.shape)
# print(y_test_onehot.shape)

n_steps,n_length,n_depth = 60,25,8

x_train = x_train.reshape(-1,n_steps,n_length,n_depth)
x_test = x_test.reshape(-1,n_steps,n_length,n_depth)

# print(x_train.shape)
# print(x_test.shape)

# model_checkpoint = keras.callbacks.ModelCheckpoint(
#     'model.h5', save_best_only=True, monitor='val_accuracy', mode='max')
csv_logger = keras.callbacks.CSVLogger('CSV_Logger.csv', append=True, separator=';')
model_checkpoint = keras.callbacks.ModelCheckpoint(filepath='model.h5',verbose=1, monitor='val_accuracy',save_weights_only=True,save_best_only=True)
early = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=0, mode='auto', baseline=None, restore_best_weights=True)
learn_rate = keras.callbacks.LearningRateScheduler(step_decay)

model = Sequential()
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=9, padding='same', kernel_initializer="he_normal",strides=2,kernel_regularizer=l1(1e-04)), \
                           input_shape=(n_steps,n_length,n_depth)))
model.add(TimeDistributed(BatchNormalization(epsilon=1e-06, momentum=0.95, weights=None)))
model.add(TimeDistributed(MaxPooling1D(pool_size=8,strides=2)))
model.add(TimeDistributed(Activation('tanh')))
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=5,padding="same",kernel_initializer="he_normal",strides=2,kernel_regularizer=l1(1e-04))))
model.add(TimeDistributed(BatchNormalization(epsilon=1e-06, momentum=0.95, weights=None)))
model.add(TimeDistributed(Activation('tanh')))
model.add(TimeDistributed(Dropout(0.2093)))
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=5,padding="same",kernel_initializer="he_normal",strides=2,kernel_regularizer=l1(1e-04))))
model.add(TimeDistributed(BatchNormalization(epsilon=1e-06, momentum=0.95, weights=None)))
model.add(TimeDistributed(Activation('tanh')))
model.add(TimeDistributed(Dropout(0.2)))
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3,padding="same",kernel_initializer="he_normal",strides=2,kernel_regularizer=l1(1e-04))))
model.add(TimeDistributed(BatchNormalization(epsilon=1e-06, momentum=0.95, weights=None)))
model.add(TimeDistributed(Activation('tanh')))
model.add(TimeDistributed(Dropout(0.2)))
model.add(TimeDistributed(Activation('relu')))
model.add(TimeDistributed(Flatten()))
# model.add(Flatten())
model.add(Bidirectional(LSTM(200,return_sequences=True)))
model.add(Dropout(0.2093))
model.add(Bidirectional(LSTM(200,return_sequences=True)))
model.add(Dropout(0.2093))
# model.add(Bidirectional(LSTM(200,return_sequences=True)))
# model.add(Dropout(0.3))
# model.add(Bidirectional(LSTM(200,return_sequences=True)))
# model.add(Dropout(0.3))
# model.add(Bidirectional(SimpleRNN(200,return_sequences=True)))
# model.add(Dropout(0.3))
# model.add(Bidirectional(SimpleRNN(200,return_sequences=True)))
# model.add(Dropout(0.3))
# model.add(BatchNormalization(epsilon=1e-05, momentum=0.9, weights=None))
# model.add(Bidirectional(SimpleRNN(200)))
# model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(512, activation='tanh'))
# model.add(Dense(200, activation='tanh'))
model.add(BatchNormalization(epsilon=1e-05, momentum=0.9, weights=None))
model.add(Dense(6, activation='softmax'))

adam=optimizers.Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])
print(model.summary())

history = model.fit(x_train, y_train_onehot, epochs=200, batch_size=8, validation_data=(x_test, y_test_onehot), verbose=1, callbacks=[model_checkpoint,csv_logger,early,learn_rate])

# model.save("model.keras")

model.load_weights("model.h5")
loss,accuracy = model.evaluate(x_test, y_test_onehot, batch_size=16, verbose=1)

print("Model Accuracy(Validation) = ", accuracy)
print("Model Loss(Validation) = ", loss)

DB1a=plt.figure(figsize=(10, 8),edgecolor='black')
plt.plot(history.history['accuracy'], 'r', label='Accuracy of training data')
plt.plot(history.history['val_accuracy'], 'b', label='Accuracy of validation data')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Training Epoch')

# plt2.ylim(0)
# plt2.rcParams['figure.facecolor'] = 'white'

# ax = fig.add_axes((left, bottom, width, height))
# ax.patch.set_alpha(1.0)
plt.ylim(0)
plt.legend()
plt.show()