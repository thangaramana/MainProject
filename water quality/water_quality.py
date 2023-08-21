
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow.keras as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import model_selection
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn import tree

# Importing the dataset
dataset = pd.read_csv('water_quality.csv')
dataset=dataset.dropna(how="any")
#dataset = dataset.head(1000)
print(dataset)
#dataset.to_csv('water_quality.csv')

dataset['ammonia'] =dataset['ammonia'].astype('float')

dataset['is_safe'] =dataset['is_safe'].astype('float')

dataset.info()

print(dataset.info())



#histogram of output
plt.figure(figsize=(10,8))
plt.title("Histogram of output")
plt.hist(dataset['is_safe'],rwidth=0.9)
plt.show()




X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 101)

import tensorflow as tf
from tensorflow.keras.layers import LSTM
def create_model():
    #input layer of model for brain signals
    inputs = tf.keras.Input(shape=(x_train.shape[1],))
    #Hidden Layer for Brain signal using CNN
    expand_dims = tf.expand_dims(inputs, axis=2)

    conv1 = tf.keras.layers.Conv1D(filters=16, kernel_size=11, padding='same', activation='relu')(expand_dims)
    mxp = tf.keras.layers.MaxPooling1D(pool_size=2)(conv1)
    conv2 = tf.keras.layers.Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')(mxp)
    mxp2 = tf.keras.layers.MaxPooling1D(pool_size=2)(conv2)
    conv3 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(mxp2)
    conv4 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(conv3)
    conv5 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(conv4)
    mxp3 = tf.keras.layers.MaxPooling1D(pool_size=2)(conv5)

    #Flatten BiLSTM layer into vector form (one Dimensional array)
    bilstm = tf.keras.layers.Bidirectional(LSTM(32))(mxp3)
    #output latyer of Model
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(bilstm)


    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    print(model.summary())
    return model

#cretaing model
cnnmodel = create_model()
#Compiling model 
cnnmodel.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

#Training and Evaluting model
history = cnnmodel.fit(x_train, y_train, epochs = 30, validation_split=0.3)
loss, acc = cnnmodel.evaluate(x_test, y_test)

#Plotting Graph of Lstm model Training, Loss and Accuracy
plt.style.use("fivethirtyeight")
plt.figure(figsize = (20,6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Model Loss",fontsize=20)
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train loss', 'validation loss'], loc ='best')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Model Accuracy",fontsize=20)
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['training accuracy', 'validation accuracy'], loc ='best')
plt.show()

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import model_selection
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score

ypred = cnnmodel.predict(x_test)
ypred = ypred.round()
ypred

import seaborn as sns
#confussion Matrix
cm = confusion_matrix(y_test, ypred)
print("Confussion Matrix for SVM")
print(cm)


cm_df = pd.DataFrame(cm,
                     index = ['0','1'], 
                     columns = ['0','1'])
#Plotting the confusion matrix
plt.figure(figsize=(5,4))
sns.heatmap(cm_df, cmap=plt.cm.Blues, annot=True, fmt='d')
plt.title('Confusion Matrix of ALEXNET+BILSTM')
plt.show()


accscore = accuracy_score(y_test, ypred)

print("ALEXNET+BILSTM accuracy is ")
print(accscore)
print("")

testy = y_test
yhat_classes = ypred
precision = precision_score(testy, yhat_classes, average = 'micro')
print('Precision: %f' % precision)
recall = recall_score(testy, yhat_classes, average = 'micro')
print('Recall: %f' % recall)
f1 = f1_score(testy, yhat_classes, average = 'micro')
print('F1 score: %f' % f1)
 
# kappa
kappa = cohen_kappa_score(testy, yhat_classes)
print('Cohens kappa: %f' % kappa)

cnnmodel.save('alexnet+bilstm.h5')
