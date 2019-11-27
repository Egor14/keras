import pandas as pd
import numpy as np
import keras
from sklearn.preprocessing import StandardScaler, PowerTransformer
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
from keras.layers import Dropout
import settings_local as SETTINGS

prepared_data =  SETTINGS.DATA  + "/COORDINATES_Pred_Term.csv"
TERM_MODEL_PATH0 = SETTINGS.MODEL + '/term#0.h5'
TERM_MODEL_PATH1 = SETTINGS.MODEL + '/term#1.h5'
TERM_MODEL_PATH2 = SETTINGS.MODEL + '/term#2.h5'


def Model_0(data: pd.DataFrame):
    # 0
    ds0 = data[(data.price < data.price.quantile(0.1))]
    print('Data #0 length: ', ds0.shape)

    # Split
    maxPrice = ds0["price"].max()
    ds0["price"] = ds0["price"] / maxPrice
    X_train = ds0.drop(['term'], axis=1).values
    y_train = ds0[['term']].values

    # StandScaling
    pt_X = PowerTransformer(method='yeo-johnson', standardize=False)
    pt_y = PowerTransformer(method='yeo-johnson', standardize=False)
    sc_y = StandardScaler()
    sc_X = StandardScaler()
    #y_train = sc_y.fit_transform(y_train)
    X_train = sc_X.fit_transform(X_train)

    model = Sequential()
    # Adding the input layer and first hidden layer
    model.add(Dense(units=250, kernel_initializer='random_uniform', activation='tanh',
                    input_dim=X_train.shape[1]))
    # Add the second hidden layer
    model.add(Dense(units=250, kernel_initializer='random_uniform', activation='tanh'))
    # Add the second hidden layer

    model.add(Dense(units=10, kernel_initializer='random_uniform', activation='relu'))
    # The output layer
    model.add(Dense(units=1, kernel_initializer='random_uniform', activation='elu'))
    # print(model.summary())
    opt = keras.optimizers.Adam(lr=0.0015)  # , beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=opt, loss='mean_squared_logarithmic_error', metrics=['mse'])
    # Fitting the ANN to the training set
    model_filepath = TERM_MODEL_PATH0

    checkpoint = ModelCheckpoint(model_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    model.fit(X_train, y_train, validation_split=0.07, batch_size=16, epochs=7, callbacks=[checkpoint])
    '''
    model_json = model.to_json()
    with open("keras_weights/model0.json", "w") as json_file:
        json_file.write(model_json)
    '''
    # serialize weights to HDF5
    model.save(model_filepath)


def Model_1(data: pd.DataFrame):
    # 1
    ds1 = data[((data.price >= data.price.quantile(0.1))&(data.price <= data.price.quantile(0.8)))]
    print('Data #1 length: ', ds1.shape)

    # Split
    maxPrice = ds1["price"].max()
    ds1["price"] = ds1["price"] / maxPrice
    X_train = ds1.drop(['term'], axis=1).values
    y_train = ds1[['term']].values

    # StandScaling
    pt_X = PowerTransformer(method='yeo-johnson', standardize=False)
    pt_y = PowerTransformer(method='yeo-johnson', standardize=False)
    sc_y = StandardScaler()
    sc_X = StandardScaler()
    #y_train = sc_y.fit_transform(y_train)
    X_train = sc_X.fit_transform(X_train)

    model = Sequential()
    # Adding the input layer and first hidden layer
    model.add(Dense(units=250, kernel_initializer='random_uniform', activation='tanh',
                    input_dim=X_train.shape[1]))
    # Add the second hidden layer
    model.add(Dense(units=250, kernel_initializer='random_uniform', activation='tanh'))
    # Add the second hidden layer

    model.add(Dense(units=10, kernel_initializer='random_uniform', activation='relu'))
    # The output layer
    model.add(Dense(units=1, kernel_initializer='random_uniform', activation='elu'))
    # print(model.summary())
    opt = keras.optimizers.Adam(lr=0.0015)  # , beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=opt, loss='mean_squared_logarithmic_error', metrics=['mse'])
    # Fitting the ANN to the training set
    model_filepath = TERM_MODEL_PATH1
    checkpoint = ModelCheckpoint(model_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    model.fit(X_train, y_train, validation_split=0.07, batch_size=16, nb_epoch=7, callbacks=[checkpoint])
    '''
    model_json = model.to_json()
    with open("keras_weights/model1.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    '''
    model.save(model_filepath)

def Model_2(data: pd.DataFrame):
    # 2
    ds2 = data[((data.price > data.price.quantile(0.8)))]
    print('Data #2 length: ', ds2.shape)

    # Split
    maxPrice = ds2["price"].max()
    ds2["price"] = ds2["price"] / maxPrice
    X_train = ds2.drop(['term'], axis=1).values
    y_train = ds2[['term']].values

    # StandScaling
    pt_X = PowerTransformer(method='yeo-johnson', standardize=False)
    pt_y = PowerTransformer(method='yeo-johnson', standardize=False)
    sc_y = StandardScaler()
    sc_X = StandardScaler()
    #y_train = sc_y.fit_transform(y_train)
    X_train = sc_X.fit_transform(X_train)

    model = Sequential()
    # Adding the input layer and first hidden layer
    model.add(Dense(units=250, kernel_initializer='random_uniform', activation='tanh',
                    input_dim=X_train.shape[1]))
    # Add the second hidden layer
    model.add(Dense(units=250, kernel_initializer='random_uniform', activation='tanh'))
    # Add the second hidden layer

    model.add(Dense(units=10, kernel_initializer='random_uniform', activation='relu'))
    # The output layer
    model.add(Dense(units=1, kernel_initializer='random_uniform', activation='elu'))
    # print(model.summary())
    opt = keras.optimizers.Adam(lr=0.0015)  # , beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=opt, loss='mean_squared_logarithmic_error', metrics=['mse'])
    # Fitting the ANN to the training set
    model_filepath = TERM_MODEL_PATH2

    checkpoint = ModelCheckpoint(model_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    model.fit(X_train, y_train, validation_split=0.07, batch_size=16, nb_epoch=7, callbacks=[checkpoint])
    '''
    model_json = model.to_json()
    with open("keras_weights/model2.json", "w") as json_file:
        json_file.write(model_json)
        '''
    # serialize weights to HDF5
    model.save(model_filepath)


def TrainModel_Term():
    # Read data
    prepared_data = SETTINGS.DATA  + "/COORDINATES_Pred_Term.csv"
    data = pd.read_csv(prepared_data)
    data = data.iloc[:-100]
    Model_0(data=data)

    Model_1(data=data)

    Model_2(data=data)

TrainModel_Term()