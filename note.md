# bare bruke data fra november 11

from scipy import stats
from sklearn.preprocessing import LabelEncoder

# Encode embeddings
le = LabelEncoder()

dfTrainData[1] = le.fit_transform(dfTrainData[1])
dfTrainData

# linear regression
x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]
slope, intercept, r, p, std_err = stats.linregress(x, y)

def myfunc(x):
  return slope * x + intercept

mymodel = list(map(myfunc, x))

plt.scatter(x, y)
plt.plot(x, mymodel)
plt.show()


# Plotting
plt.figure()
plt.scatter(dfWFI['TIMESTAMP'], dfSolution['POWER'], color='orange', label='data')
plt.plot(dfWFI['TIMESTAMP'], SVR_y_pred, color='b', label='prediction') 
plt.xlabel('Date')
plt.ylabel('Power')
plt.xticks(dfWFI['TIMESTAMP'][::79],  rotation='vertical')
plt.legend()
plt.show()

# Convert the 'TIMESTAMP' column to datetime format
dfTrainData['TIMESTAMP'] = pd.to_datetime(dfTrainData['TIMESTAMP'])

# Filter the DataFrame to include only the data for November 2013
dfTrainData = dfTrainData[dfTrainData['TIMESTAMP'].dt.month == 11]
dfTrainData


# Convert TIMESTAMP (string) to float
year = []
month = []
day = []
time = []

# converting dfTrainData
for index, row in dfTrainData.iterrows():
    timestamp = datetime.strptime(row['TIMESTAMP'],  "%Y%m%d %H:%M")
    year.append(timestamp.year)
    month.append(timestamp.month)
    day.append(timestamp.day)
    time.append(timestamp.hour)

dfTrainData['YEAR'] = year
dfTrainData['MONTH'] = month
dfTrainData['DAY'] = day
dfTrainData['TIME'] = time
dfTrainData = dfTrainData.drop(columns=['TIMESTAMP'])

year = []
month = []
day = []
time = []

# converting dfWFI
for index, row in dfWFI.iterrows():
    timestamp = datetime.strptime(row['TIMESTAMP'],  "%Y%m%d %H:%M")
    year.append(timestamp.year)
    month.append(timestamp.month)
    day.append(timestamp.day)
    time.append(timestamp.hour)

dfWFI['YEAR'] = year
dfWFI['MONTH'] = month
dfWFI['DAY'] = day
dfWFI['TIME'] = time
dfWFI = dfWFI.drop(columns=['TIMESTAMP'])


conda install pytorch torchvision torchaudio cpuonly -c pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

""" Artificial Neural Networks (ANN) """
# Defining the model 
ANN_model = keras.Sequential([ 
    keras.layers.Dense(4, input_shape=(1,), activation='relu'), 
    keras.layers.Dense(4, activation='sigmoid') 
]) 

# Defining loss function
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

# Compiling the model 
ANN_model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=['accuracy']) 
  
# Train the model
ANN_model.fit(X_train, y_train, epochs=10, batch_size=8, verbose=0) 

# Prediction
ANN_y_pred = ANN_model.predict(X_test)

# Evaluate the model
results = ANN_model.evaluate(X_test, dfSolution['POWER'], batch_size=8)
print('Test loss, Test accuracy:', results)
print('RMSE for Artificial Neural Network:', results[0])


X_train = X_train.reshape(-1, 1)