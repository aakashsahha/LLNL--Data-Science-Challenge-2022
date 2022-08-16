import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,Dropout
import os
import keras
from ctgan import CTGANSynthesizer



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import tensorflow_addons as tfa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score


from tensorflow.keras.optimizers import Adam, RMSprop, SGD

from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Add, Dense, Activation, Dropout, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import random_uniform, glorot_uniform, constant, identity

# simple example of how to read the data files

# read task 1 data
data_csv = "DSC_2022/gitlab/data/mpro_exp_data2_rdkit_feat.csv"
data_df = pd.read_csv(data_csv)

data_df.drop(columns= ['lib_name'], inplace=True)
data_df.replace([np.inf, -np.inf], np.nan, inplace=True)
data_df.dropna(inplace=True)


print(data_df)

for column in data_df:
    if data_df[column].dtypes == 'float64':
        if (data_df[column].quantile(0.75) - (data_df[column].quantile(0.25))) == 0:
            data_df.drop(columns=[column],inplace=True)
        else:
            data_df[column] = (data_df[column] - data_df[column].median())/(data_df[column].quantile(0.75) - data_df[column].quantile(0.25))

# Check if the dataset is imbalanced
print(data_df)
print(print(data_df.groupby(['label','subset']).size()))


"""Task 1:"""

df_test = data_df[data_df['subset'] == 'test']
df_train = data_df[data_df['subset'] == 'train']
df_valid = data_df[data_df['subset'] == 'valid']

"""Training:"""

# The train matrix consist of 208 feature vectors
X_train = df_train[df_train.columns[5:]]
X_test = df_test[df_test.columns[5:]]
X_valid = df_valid[df_test.columns[5:]]


Y_train = df_train[df_train.columns[3]]
Y_test = df_test[df_test.columns[3]]
Y_valid = df_valid[df_valid.columns[3]]





model = Sequential([
    Dense(64, activation='relu', input_shape=(203,)),
    Dense(256, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform()),
    Dropout(0.2),
    Dense(512, activation='relu', kernel_initializer=tf.keras.initializers.GlorotUniform()),
    Dropout(0.3),
    Dense(1, activation='sigmoid'),
])

print(model.summary())
callbacks = [
        keras.callbacks.ModelCheckpoint(
        "best_model_"+model._name+".h5", save_best_only=True, monitor="val_accuracy", mode = 'max', verbose = 1,
        ),
    ]

adam_fine = RMSprop(learning_rate=0.00005)
model.compile(
        optimizer=adam_fine,
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(),'accuracy'],
    )

history = model.fit(
    X_train,Y_train,
    batch_size=256,
    epochs=2000,
    callbacks = callbacks,
    validation_data=(X_valid,Y_valid),
    verbose=1,
)

model.save('final_model_'+model._name+'.h5')
# Testing part
best_model = keras.models.load_model("best_model_"+model._name+".h5")

best_model.evaluate(X_test, Y_test)[1]




plt.figure(figsize=(10,10))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('accuracy per epoch')
plt.ylabel('accuracy',fontsize=12)
plt.xlabel('epoch',fontsize=12)
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('accuracy_'+model._name+'.pdf')


plt.figure(figsize=(10,10))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('loss per epoch')
plt.ylabel('loss',fontsize=12)
plt.xlabel('epoch',fontsize=12)
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('loss_'+model._name+'.pdf')

plt.figure(figsize=(10,10))
plt.plot(history.history['auc'])
plt.plot(history.history['val_auc'])
plt.xlabel('epoch',fontsize=12)
plt.title('AUC per epoch')
plt.ylabel('ROC_AUC',fontsize=12)
plt.legend(['train', 'validation'], loc='lower right', fontsize ='large')
plt.savefig('ROC_'+model._name+'.pdf')