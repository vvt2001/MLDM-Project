import glob
import pandas as pd
import numpy as np

# Get CSV files list from a folder
path_train = "C:/Users/DELL/Desktop/Train"
csv_files_train = glob.glob(path_train + "/*.csv")
path_test = "C:/Users/DELL/Desktop/Test"
csv_files_test = glob.glob(path_test + "/*.csv")

x_train = []
x_test = []
y_train = []
y_test = []

train_index = 1
test_index = 1

for file in csv_files_train:
    #set column name
    columns = ['gx','gy','gz','ax','ay','az']
    selected_columns = ['ax','ay','az']
    df = pd.read_csv(file, sep=',', header=None, names=columns, usecols=selected_columns)
    df = df.drop([df.index[0], df.index[1]])
    df = df.reset_index(drop=True)

    #print(df) 
    print('read: ' + str(train_index) + ' ' +  file)

    X_train = []
    Y_train = []
    Z_train = []

    for column in selected_columns:
        # Perform Fourier Transform on the 'Value' column
        fft_vals = np.fft.fft(df[column].values)

        sum = 0
        for value in fft_vals:
            sum+=np.abs(value)

        # Get the corresponding frequencies for each FFT coefficient
        freqs = np.fft.fftfreq(len(df), d=(df.index[1] - df.index[0]))

        # Create a new dataframe to store the FFT coefficients and corresponding frequencies
        fft_df = pd.DataFrame({'FFT Coefficients': np.abs(fft_vals)/sum,
                               'Frequency': freqs})

        # Remove the first FFT coefficient (DC offset)
        fft_df = fft_df[fft_df['Frequency'] > 0]

        #print(fft_df)

        # create a new frequency array with an interval of 0.001Hz
        min_freq = 0
        max_freq = round(fft_df['Frequency'].max(), 3)
        new_freq = np.arange(min_freq, max_freq + 0.002, 0.002)

        # interpolate coefficient values for new frequency data using numpy.interp()
        new_coef = np.interp(new_freq, fft_df['Frequency'], fft_df['FFT Coefficients'])

        # create a new dataframe with the interpolated frequency and coefficient data
        df_new = pd.DataFrame({'FFT Coefficients': new_coef, 'Frequency': new_freq})

        # print the new dataframe
        #print(df_new)

        #take the first ~100 values
        df_new = df_new.take(slice(0, 100))
        
        fft_array = df_new['FFT Coefficients'].to_numpy()
        #print(fft_array)

        if column == 'ax':
            X_train = fft_array
        if column == 'ay':
            Y_train = fft_array
        if column == 'az':
            Z_train = fft_array
            
        print('fft: ' + column + ' ' + str(train_index) )


    # Initialize an empty list to hold the concatenated arrays
    concatenated_array = []

    # Loop through each array and append its elements to the concatenated array
    for array in [X_train, Y_train, Z_train]:
        for element in array:
            concatenated_array.append(element)

    for i in range(len(concatenated_array)):
        if concatenated_array[i] < 0:
            concatenated_array[i] *= -1

    x_train.append(concatenated_array)
    y_train.append(file)

    train_index += 1

for file in csv_files_test:
    #set column name
    columns = ['gx','gy','gz','ax', 'ay','az']
    selected_columns = ['ax','ay','az']
    df = pd.read_csv(file, sep=',', header=None, names=columns, usecols=selected_columns)
    df = df.drop([df.index[0], df.index[1]])
    df = df.reset_index(drop=True)

    #print(df) 
    print('read: ' + str(test_index) + ' ' +  file )

    X_test = []
    Y_test = []
    Z_test = []

    for column in selected_columns:
        # Perform Fourier Transform on the 'Value' column
        fft_vals = np.fft.fft(df[column].values)
        sum = 0
        for value in fft_vals:
            sum+=np.abs(value)

        # Get the corresponding frequencies for each FFT coefficient
        freqs = np.fft.fftfreq(len(df), d=(df.index[1] - df.index[0]))

        # Create a new dataframe to store the FFT coefficients and corresponding frequencies
        fft_df = pd.DataFrame({'FFT Coefficients': np.abs(fft_vals)/sum,
                               'Frequency': freqs})

        # Remove the first FFT coefficient (DC offset)
        fft_df = fft_df[fft_df['Frequency'] > 0]

        #print(fft_df)

        # create a new frequency array with an interval of 0.001Hz
        min_freq = 0
        max_freq = round(fft_df['Frequency'].max(), 3)
        new_freq = np.arange(min_freq, max_freq + 0.002, 0.002)

        # interpolate coefficient values for new frequency data using numpy.interp()
        new_coef = np.interp(new_freq, fft_df['Frequency'], fft_df['FFT Coefficients'])

        # create a new dataframe with the interpolated frequency and coefficient data
        df_new = pd.DataFrame({'FFT Coefficients': new_coef, 'Frequency': new_freq})

        # print the new dataframe
        #print(df_new)

        #take the first ~100 values
        df_new = df_new.take(slice(0, 100))

        fft_array = df_new['FFT Coefficients'].to_numpy()
        #print(fft_array)
        if column == 'ax':
            X_test = fft_array
        if column == 'ay':
            Y_test = fft_array
        if column == 'az':
            Z_test = fft_array
            
        print('fft: ' + column + ' ' + str(test_index) )


    # Initialize an empty list to hold the concatenated arrays
    concatenated_array = []

    # Loop through each array and append its elements to the concatenated array
    for array in [X_test, Y_test, Z_test]:
        for element in array:
            concatenated_array.append(element)

    for i in range(len(concatenated_array)):
        if concatenated_array[i] < 0:
            concatenated_array[i] *= -1

    x_test.append(concatenated_array)

    test_index += 1

y_test = y_train

x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# create an instance of svm classifier with a linear kernel
clf = svm.SVC(kernel='linear')

# fit the svm classifier on the training data
clf.fit(x_train, y_train)

# use the svm classifier to predict the labels of the testing data
y_pred = clf.predict(x_test)

# compute the accuracy of the svm classifier on the testing data
accuracy = accuracy_score(y_test, y_pred)

# print the accuracy of the svm classifier
print("accuracy:", accuracy)

print("end")


######TEST GRAPH#####
##set column name
#columns = ['gx','gy','gz','ax','ay','az']
#selected_columns = ['ax','ay','az']
#df = pd.read_csv('C:/Users/DELL/Desktop/Train\T0_ID355444_Walk1.csv', sep=',', header=None, names=columns, usecols=selected_columns)
#df = df.drop([df.index[0], df.index[1]])
#df = df.reset_index(drop=True)

#print(df) 

## Perform Fourier Transform on the 'Value' column
#fft_vals = np.fft.fft(df['ax'].values)

#sum = 0
#for value in fft_vals:
#    sum+=np.abs(value)

## Get the corresponding frequencies for each FFT coefficient
#freqs = np.fft.fftfreq(len(df), d=(df.index[1] - df.index[0]))

## Create a new dataframe to store the FFT coefficients and corresponding frequencies
#fft_df = pd.DataFrame({'FFT Coefficients': np.abs(fft_vals)/sum,
#                        'Frequency': freqs})

## Remove the first FFT coefficient (DC offset)
#fft_df = fft_df[fft_df['Frequency'] > 0]

## create a new data point as a list
#new_data = [0, 0]

## insert the new data point into the DataFrame using loc[] and index label 0
#fft_df.loc[0] = new_data
#fft_df.index = fft_df.index + 1

## sort the index in ascending order
#fft_df = fft_df.sort_index()

#print(fft_df)

## create a new frequency array with an interval of 0.001Hz
#min_freq = 0
#max_freq = round(fft_df['Frequency'].max(), 3)
#new_freq = np.arange(min_freq, max_freq + 0.002, 0.002)

## interpolate coefficient values for new frequency data using numpy.interp()
#new_coef = np.interp(new_freq, fft_df['Frequency'], fft_df['FFT Coefficients'])

## create a new dataframe with the interpolated frequency and coefficient data
#new_df = pd.DataFrame({'FFT Coefficients': new_coef, 'Frequency': new_freq})

#for i in range(len(new_df)):
#    if new_df['FFT Coefficients'][i] < 0:
#        new_df['FFT Coefficients'][i] *= -1

## print the new dataframe
#print(new_df)


##create graph
#import matplotlib.pyplot as plt

## convert into x and y
#x = list(range(len(new_df.index)))
#y = new_df['FFT Coefficients']

## plot the data
#fig = plt.figure()
#plt.plot(x,y)
#plt.ylabel('FFT Coefficients')
#plt.xlabel('Frequency')
#plt.show()
