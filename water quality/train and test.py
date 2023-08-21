# STEP 1: Generate Two Large Prime Numbers (p,q) randomly
from random import randrange, getrandbits
from tkinter import *
from tkinter import ttk  
from tkinter import Menu  
from tkinter import messagebox as mbox  
# import filedialog module
from tkinter import filedialog
flg=0;
import tkinter as tk
import tkinter
from tkinter import *
from PIL import Image, ImageTk
# Create a photoimage object of the image in the path
import seaborn as sns

import tkinter as tk
from PIL import ImageTk, Image
from tkintertable import TableCanvas, TableModel
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



import numpy as np
import matplotlib.pyplot as plts
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import tensorflow.keras as tf

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score

import seaborn as sns
import pandas as pd

import pandas as pd
def train():
    print("training")
        
            
        
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

    out = "Accuracy Score : "
    out += str(accscore)
    out += "\n Precision score : "
    out += str(precision)
    out += "\n Recall score : "
    out += str(recall)
    out += "\n F1 score : "
    out += str(f1)
    out += "\n cohens kappa score : "
    out += str(kappa)
    
    app = tk.Tk()
    app.title("Scores")
    ttk.Label(app, text=out).grid(column=0,row=0,padx=20,pady=30)  
    menuBar = Menu(app)
    app.config(menu=menuBar)
    

def test():
    print("testing")
        
    import tkinter as tk
    import tkinter
    from PIL import Image, ImageTk
    from tkinter import ttk  
    from tkinter import Menu  
    from tkinter import messagebox as mbox  
    # import filedialog module
    from tkinter import filedialog
    flg=0;
    import tkinter as tk

    # Function for opening the
    # file explorer window
    def browseFiles():
        filename = filedialog.askopenfilename(initialdir = "/",
                                              title = "Select a CSV File",
                                              filetypes = (("CSV files",
                                                            "*.csv*"),
                                                           ("all files",
                                                            "*.*")))
        # Change label contents
        label_file_explorer.configure(text="File Opened: "+filename)
        global f
        f = filename


    def start():

        print("Process Started")
        dataset = pd.read_csv(f)
        dataset=dataset.dropna(how="any")
        print(dataset)

        print(dataset.info())

        X = dataset.iloc[:,:].values

        # load the model from disk
        model = tf.models.load_model("alexnet+bilstm.h5")
        ypred = model.predict(X)
        ypred = ypred.round()
        print(ypred)
        app = tk.Tk()
        if(ypred==0):
            print("Water Quality is Good")
            label_file_explorer.configure(text="Result for the Input data: Water Quality is Good")
            app.title("Water Quality Prediction system")
            ttk.Label(app, text="Result for the patient data: Water Quality is Good").grid(column=0,row=0,padx=20,pady=30)  
            menuBar = Menu(app)
            app.config(menu=menuBar)
        elif(ypred==1):
            print("Water Quality is Bad")
            label_file_explorer.configure(text="Result for the input data: Water Quality is Bad")
            app.title("Water Quality Prediction system")
            ttk.Label(app, text="Result for the Input data: Water Quality is Bad").grid(column=0,row=0,padx=20,pady=30)
            menuBar = Menu(app)
            app.config(menu=menuBar)
        
        
    window = Tk()
  
    # Set window title
    window.title('Water Quality Prediction system')
      
    # Set window size
    window.geometry("700x400")
      
    #Set window background color
    window.config(background = "white")
      
    # Create a File Explorer label
    label_file_explorer = Label(window,
                                text = "Please give Input dataset",
                                width = 100, height = 4,
                                fg = "blue")
         
    button_explore = Button(window,
                            text = "Browse Input data Files",
                            command = browseFiles)
    button_exit = Button(window,
                         text = "exit",
                         command = exit)  
    button_start = Button(window,
                         text = "Start Analyzing Data",
                         command = start)

       
    # Grid method is chosen for placing
    # the widgets at respective positions
    # in a table like structure by
    # specifying rows and columns
    label_file_explorer.grid(column = 1, row = 1, padx=5, pady=5)
    button_explore.grid(column = 1, row = 3, padx=5, pady=5)
    button_exit.grid(column = 1,row = 9, padx=5, pady=5)
    button_start.grid(column = 1,row = 12, padx=5, pady=5)
      
    # Let the window wait for any events
    
    
    window.mainloop()




if __name__ == '__main__':    
    # Create the main window
    root = tk.Tk()
    root.title("My GUI")
    root.geometry("500x300")

    # Load the background image
    bg_image = ImageTk.PhotoImage(Image.open("bg.jpg"))

    # Load the image to be displayed
    image = Image.open("client.jpg")
    resized_image = image.resize((90, 90))
    tk_image = ImageTk.PhotoImage(resized_image)

    # Create a canvas to display the background image
    canvas = tk.Canvas(root, width=500, height=300)
    canvas.pack(fill="both", expand=True)
    canvas.create_image(0, 0, image=bg_image, anchor="nw")

    # Create a title in the center of the GUI with a bigger font
    title = tk.Label(canvas, text="Water Quality Prediction System", font=("Arial", 24))
    title.place(relx=0.5, rely=0.1, anchor="center")

    # Display the image in the center of the GUI
    image_label = tk.Label(canvas, image=tk_image)
    image_label.place(relx=0.5, rely=0.4, anchor="center")

    # Create two big buttons for Train and Test
    train_button = tk.Button(canvas, text="Train", bg="green", fg="white", font=("Arial", 20), command = train)
    train_button.place(relx=0.3, rely=0.7, anchor="center")
    test_button = tk.Button(canvas, text="Test", bg="red", fg="white", font=("Arial", 20), command= test)
    test_button.place(relx=0.7, rely=0.7, anchor="center")

    # Start the GUI
    root.mainloop()

