import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# DATA PROCESSING

dataset = pd.read_csv('life_expectancy.csv')

#we need general worldwide predictive model 
#not specific to country so we dorp it
dataset = dataset.drop(['Country'], axis = 1)

#as aout prediction is about age, 
#our label data will be Life expectancy column
labels = dataset.iloc[:, -1]

#everything else goes to features
features = dataset.iloc[:, 0:-1]

#convert all the categorical columns into numerical
features = pd.get_dummies(features)

#print(features.head())
#print(dataset.describe())
#split data for training/testing 80/20
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.20, random_state=23)

#normilize data
numerical_features = features.select_dtypes(include=['float64', 'int64'])
numerical_columns = numerical_features.columns
ct = ColumnTransformer([("only numeric", StandardScaler(), numerical_columns)], remainder='passthrough')
features_train_scaled = ct.fit_transform(features_train)
features_test_scaled = ct.transform(features_test)

#building NN model
my_model = Sequential()
input = InputLayer(input_shape = (features.shape[1], ))
my_model.add(input)
my_model.add(Dense(64, activation = "relu"))
my_model.add(Dense(1))
#print(my_model.summary())
opt = Adam(learning_rate = 0.01)
my_model.compile(loss = 'mse', metrics = ['mae'], optimizer = opt)

#Train NN model and evaluate, print final loss and metrics
my_model.fit(features_train, labels_train, epochs = 20, batch_size = 5, verbose = 0)
res_mse, res_mae = my_model.evaluate(features_test, labels_test, verbose = 0)
print(res_mse, res_mae)
