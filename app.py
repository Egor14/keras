from flask import Flask, request, jsonify
from joblib import dump, load
import keras
import pandas as pd
import math as m
import json
import numpy as np
#import Realty.realty_main.Tensorflow_PyTorch.SERVER_KERAS.settings_local_keras as SETTINGS
from sklearn.preprocessing import StandardScaler
from keras import backend as K
import settings_local as SETTINGS

app = Flask(__name__)

prepared_data = SETTINGS.DATA + "/COORDINATES_Pred_Term.csv"
data = pd.read_csv(prepared_data)


sc_y = StandardScaler()
sc_X = StandardScaler()



def func_pred_price0(params: list):
    model_price = keras.models.load_model(SETTINGS.PRICE_MODEL_PATH0)
    X = params
    X_test = np.asarray(params).reshape(1, 12)
    X_test = sc_X.fit_transform(X_test)
    y_pred = model_price.predict(X_test)
    print(type(y_pred))
    K.clear_session()
    return y_pred


def func_pred_price1(params: list):
    model_price = keras.models.load_model(SETTINGS.PRICE_MODEL_PATH1)
    X = params
    X_test = np.asarray(params).reshape(1, 12)
    X_test = sc_X.fit_transform(X_test)
    y_pred = model_price.predict(X_test)
    print(type(y_pred))
    K.clear_session()
    return y_pred


def func_pred_price2(params: list):
    model_price = keras.models.load_model(SETTINGS.PRICE_MODEL_PATH2)
    X = params
    X_test = np.asarray(params).reshape(1, 12)
    X_test = sc_X.fit_transform(X_test)
    y_pred = model_price.predict(X_test)
    print(type(y_pred))
    K.clear_session()
    return y_pred


def func_pred_term0(params: list):
    model_price = keras.models.load_model(SETTINGS.TERM_MODEL_PATH0)
    X = params
    X_test = np.asarray(params).reshape(1, 13)
    X_test = sc_X.fit_transform(X_test)
    y_pred = model_price.predict(X_test)
    K.clear_session()
    return y_pred


def func_pred_term1(params: list):
    model_price = keras.models.load_model(SETTINGS.TERM_MODEL_PATH1)
    X = params
    X_test = np.asarray(params).reshape(1, 13)
    X_test = sc_X.fit_transform(X_test)
    y_pred = model_price.predict(X_test)
    K.clear_session()
    return y_pred


def func_pred_term2(params: list):
    model_price = keras.models.load_model(SETTINGS.TERM_MODEL_PATH2)
    X = params
    X_test = np.asarray(params).reshape(1, 13)
    X_test = sc_X.fit_transform(X_test)
    y_pred = model_price.predict(X_test)
    K.clear_session()
    return y_pred

@app.route('/')
def main():
   return 'Hi :)'


@app.route('/map')
def map():
    building_type_str = request.args.get('building_type_str')
    longitude = float(request.args.get('lng'))
    latitude = float(request.args.get('lat'))
    full_sq = float(request.args.get('full_sq'))
    kitchen_sq = float(request.args.get('kitchen_sq'))
    life_sq = request.args.get('life_sq')
    is_apartment = int(request.args.get('is_apartment'))
    renovation = int(request.args.get('renovation'))
    has_elevator = int(request.args.get('has_elevator'))
    floor_first = int(request.args.get('floor_first'))
    floor_last = int(request.args.get('floor_last'))
    time_to_metro = int(request.args.get('time_to_metro'))
    X = (m.cos(latitude) * m.cos(longitude))
    Y = (m.cos(latitude) * m.sin(longitude))

    list_of_requested_params_price = [renovation, has_elevator, longitude, latitude, full_sq, kitchen_sq,
                                      is_apartment, time_to_metro, floor_last, floor_first, X, Y]
    # Data
    price = 0
    data = pd.read_csv(SETTINGS.DATA)

    if full_sq < float(data.full_sq.quantile(0.1)):
        print('0')
        ds0 = data[(data.full_sq < data.full_sq.quantile(0.1))]
        maxPrice = ds0["price"].max()
        price = func_pred_price0(list_of_requested_params_price)
        price = abs(price[0][0] * maxPrice)
        K.clear_session()
        print(price)
    elif ((full_sq >= float(data.full_sq.quantile(0.1))) & (full_sq <= float(data.full_sq.quantile(0.8)))):
        print('1')
        ds1 = data[((data.full_sq >= data.full_sq.quantile(0.1)) & (data.full_sq <= data.full_sq.quantile(0.8)))]
        maxPrice = ds1["price"].max()
        price = func_pred_price1(list_of_requested_params_price)
        price = abs(price[0][0] * maxPrice)
        K.clear_session()
        print(price)
    elif full_sq > float(data.full_sq.quantile(0.8)):
        print('2')
        ds2 = data[((data.full_sq > data.full_sq.quantile(0.8)))]
        maxPrice = ds2["price"].max()
        price = func_pred_price2(list_of_requested_params_price)
        price = abs(price[0][0] * maxPrice)
        K.clear_session()
        print(price)
    price_meter_sq = price / full_sq

    # SALE TERM PREDICTION
    list_of_requested_params_term = [renovation, has_elevator, longitude, latitude, np.log1p(price), full_sq, kitchen_sq,
                                     is_apartment, time_to_metro,
                                     floor_last, floor_first, X, Y]
    term = 0
    # Data
    data = pd.read_csv(SETTINGS.DATA)
    data['price_meter_sq'] = data[['price', 'full_sq']].apply(
        lambda row: (row['price'] /
                     row['full_sq']), axis=1)
    if float(price) < float(data.price.quantile(0.2)):
        print('0')
        term = func_pred_term0(list_of_requested_params_term)
        K.clear_session()

    elif (float(price) >= float(data.price.quantile(0.2))) & (float(price) <= float(data.price.quantile(0.85))):
        print('1')
        term = func_pred_term1(list_of_requested_params_term)
        K.clear_session()

    elif float(price) > float(data.price.quantile(0.85)):
        print('2')
        term = func_pred_term2(list_of_requested_params_term)
        K.clear_session()

    filter1 = ((data.full_sq <= full_sq + 1) & (
            (data.longitude >= longitude - 0.01) & (data.longitude <= longitude + 0.01) &
            (data.latitude >= latitude - 0.01) & (data.latitude <= latitude + 0.01)) &
               ((data.price_meter_sq <= price_meter_sq + 3000) & (data.price_meter_sq >= price_meter_sq - 3000))
               & (data.term < 400) & (
                           (data.time_to_metro >= time_to_metro - 2) & (data.time_to_metro <= time_to_metro + 2)))
    ds = data[filter1]
    print(ds.shape)

    x = ds.term.tolist()
    y = ds.price.tolist()
    a = []
    a += ({'x{0}'.format(k): x, 'y{0}'.format(k): y} for k, x, y in zip(list(range(len(x))), x, y))

    return jsonify({'Price': str(price), 'Duration': term.tolist()[0], 'PLot': list(a)})
    # , 'Term': term})
    # return 'Price {0} \n Estimated Sale Time: {1} days'.format(price, term)

