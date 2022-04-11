import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import datetime
import os
from sklearn import metrics
import math
import matplotlib.pyplot as plt
import random as rn
from statsmodels.tsa.seasonal import seasonal_decompose, STL

import tensorflow as tf
from tensorflow.keras import Model, optimizers
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, Lambda, dot, Activation, concatenate, Layer
from tensorflow.keras.layers import Concatenate, Input, Permute, Reshape, Conv1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K

from skopt import gp_minimize
from skopt.utils import use_named_args
from skopt.space import Real, Categorical, Integer

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True, \
                            intra_op_parallelism_threads=1, inter_op_parallelism_threads=1))
tf.compat.v1.keras.backend.set_session(sess)
tf.compat.v1.disable_eager_execution()

prj_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(prj_dir, 'datasets')
model_dir = os.path.join(prj_dir, 'models_DH')
result_dir = os.path.join(prj_dir, 'results_DH')
image_dir = os.path.join(prj_dir, 'images_DH')

if not os.path.isdir(model_dir):
    os.makedirs(model_dir)
if not os.path.isdir(result_dir):
    os.makedirs(result_dir)
if not os.path.isdir(image_dir):
    os.makedirs(image_dir)

test_name = 'ther-git'

# =================================================================================================================

timesteps = 24

dim_learning_rate = Real(low=1e-3, high=1e-2, prior='log-uniform', name='learning_rate')
dim_filter_val = Integer(low=64, high=128, name='filter_val', dtype=int)
dim_kernel_val = Integer(low=2, high=5, name='kernel_val', dtype=int)
dim_num_lstm_nodes = Integer(low=16, high=64, name='num_lstm_nodes', dtype=int)
dim_activation = Categorical(categories=['tanh', 'relu'], name='activation')
dim_batch_size = Integer(low=1000, high=5000, name='batch_size', dtype=int)

dimensions = [dim_learning_rate, dim_filter_val, dim_kernel_val, dim_num_lstm_nodes, dim_activation, dim_batch_size]
default_parameters = [1e-2, 128, 3, 64, 'relu', 2000]

# =================================================================================================================
def data_thermaluse(df):

    date_index = pd.date_range(start='2011-01-01 01:00:00', end='2021-01-17 14:00:00', freq='1H')
    df.sort_index(inplace=True)
    df = df.loc['2011-01-01 01:00:00':'2021-01-17 14:00:00']
    df = df.reindex(date_index, fill_value=np.nan)

    # ---------------------------------------------------------------------------------------------------------
    ts = df.LOAD

    result = seasonal_decompose(ts, model='additive')
    df_sd = pd.concat([result.seasonal, result.trend], axis=1)
    df_sd = pd.concat([df_sd, result.resid], axis=1)
    df = pd.concat([df, df_sd], axis=1)
    df.rename(columns={'seasonal': 'seasonal_sd', 'trend': 'trend_sd', 'resid': 'resid_sd'}, inplace=True)

    result = STL(ts, seasonal=25).fit()
    df_sd = pd.concat([result.seasonal, result.trend], axis=1)
    df_sd = pd.concat([df_sd, result.resid], axis=1)
    df = pd.concat([df, df_sd], axis=1)
    df.rename(columns={'seasonal': 'seasonal_stl', 'trend': 'trend_stl'}, inplace=True)

    df_skew = ts.rolling(window=6, min_periods=3).skew().to_frame()
    df_skew.rename(columns={'LOAD': 'LOAD_skew'}, inplace=True)
    df = pd.concat([df, df_skew], axis=1)

    df_kurt = ts.rolling(window=6, min_periods=3).kurt().to_frame()
    df_kurt.rename(columns={'LOAD': 'LOAD_kurt'}, inplace=True)
    df = pd.concat([df, df_kurt], axis=1)

    # ---------------------------------------------------------------------------------------------------------
    for wd in [6, 13, 20]:
        for dtid, dtval in df.iterrows():
            try:
                df.at[dtid, 'LOAD_{}d_+3'.format(wd)] = \
                    df.at[(dtid - datetime.timedelta(days=wd) + datetime.timedelta(hours=3)), 'LOAD']
                df.at[dtid, 'LOAD_{}d_+2'.format(wd)] = \
                    df.at[(dtid - datetime.timedelta(days=wd) + datetime.timedelta(hours=2)), 'LOAD']
                df.at[dtid, 'LOAD_{}d_+1'.format(wd)] = \
                    df.at[(dtid - datetime.timedelta(days=wd) + datetime.timedelta(hours=1)), 'LOAD']
                df.at[dtid, 'LOAD_{}d_0'.format(wd)] = \
                    df.at[(dtid - datetime.timedelta(days=wd)), 'LOAD']
                df.at[dtid, 'LOAD_{}d_-1'.format(wd)] = \
                    df.at[(dtid - datetime.timedelta(days=wd) - datetime.timedelta(hours=1)), 'LOAD']
                df.at[dtid, 'LOAD_{}d_-2'.format(wd)] = \
                    df.at[(dtid - datetime.timedelta(days=wd) - datetime.timedelta(hours=2)), 'LOAD']
                df.at[dtid, 'LOAD_{}d_-3'.format(wd)] = \
                    df.at[(dtid - datetime.timedelta(days=wd) - datetime.timedelta(hours=3)), 'LOAD']

                df.at[dtid, 'Ratio_{}d_+3'.format(wd)] = \
                    df.at[dtid, 'LOAD'] / df.at[(dtid - datetime.timedelta(days=wd) + datetime.timedelta(hours=3)), 'LOAD']
                df.at[dtid, 'Ratio_{}d_+2'.format(wd)] = \
                    df.at[dtid, 'LOAD'] / df.at[(dtid - datetime.timedelta(days=wd) + datetime.timedelta(hours=2)), 'LOAD']
                df.at[dtid, 'Ratio_{}d_+1'.format(wd)] = \
                    df.at[dtid, 'LOAD'] / df.at[(dtid - datetime.timedelta(days=wd) + datetime.timedelta(hours=1)), 'LOAD']
                df.at[dtid, 'Ratio_{}d_0'.format(wd)] = \
                    df.at[dtid, 'LOAD'] / df.at[(dtid - datetime.timedelta(days=wd)), 'LOAD']
                df.at[dtid, 'Ratio_{}d_-1'.format(wd)] = \
                    df.at[dtid, 'LOAD'] / df.at[(dtid - datetime.timedelta(days=wd) - datetime.timedelta(hours=1)), 'LOAD']
                df.at[dtid, 'Ratio_{}d_-2'.format(wd)] = \
                    df.at[dtid, 'LOAD'] / df.at[(dtid - datetime.timedelta(days=wd) - datetime.timedelta(hours=2)), 'LOAD']
                df.at[dtid, 'Ratio_{}d_-3'.format(wd)] = \
                    df.at[dtid, 'LOAD'] / df.at[(dtid - datetime.timedelta(days=wd) - datetime.timedelta(hours=3)), 'LOAD']
            except:
                pass

    # ---------------------------------------------------------------------------------------------------------
    df['YEAR'] = df.index.year
    df['MONTH'] = df.index.month
    df['DAY'] = df.index.day
    df['HOUR'] = df.index.hour
    df.index.name = 'DT'

    # ---------------------------------------------------------------------------------------------------------
    for col in df.columns:
        periods = [2, 4, 6, 8, 10, 12, 24]
        for rp in periods:
            df[col].fillna(df[col].rolling(rp, min_periods=1).mean(), inplace=True)

    return df

# =================================================================================================================
def data_weather():

    df = pd.read_csv(os.path.join(data_dir, 'WEATHER_FORE-20110102-20210117.csv'), \
                     engine='python', index_col='DT', parse_dates=True, encoding='utf-8')
    df.sort_index(inplace=True)

    # ---------------------------------------------------------------------------------------------------------
    for rp in range(1, 24):
        df['TEMPERATURE'].fillna(df['TEMPERATURE'].rolling(rp, min_periods=1).mean(), inplace=True)
        df['WINDS_PEED'].fillna(df['WINDS_PEED'].rolling(rp, min_periods=1).mean(), inplace=True)
        df['HUMIDITY'].fillna(df['HUMIDITY'].rolling(rp, min_periods=1).mean(), inplace=True)

        df['TEMPERATURE'].fillna(df['TEMPERATURE'].shift(rp), inplace=True)
        df['WINDS_PEED'].fillna(df['WINDS_PEED'].shift(rp), inplace=True)
        df['HUMIDITY'].fillna(df['HUMIDITY'].shift(rp), inplace=True)

    df.fillna(0.0, inplace=True)

    # ---------------------------------------------------------------------------------------------------------
    df['YEAR'] = df.index.year
    df['MONTH'] = df.index.month
    df['DAY'] = df.index.day

    dfg = df.groupby(by=['YEAR', 'MONTH', 'DAY'])['PRECIPITATION', 'SNOWFALL', 'HUMIDITY'].cumsum().reset_index()
    dfg.sort_values(by=['DT'], inplace=True)
    dfg.set_index('DT', drop=True, inplace=True)
    dfg.rename(columns={'PRECIPITATION': 'precipitation_cumsum', 'SNOWFALL': 'snowfall_cumsum',
                        'HUMIDITY': 'humidity_cumsum'}, inplace=True)
    df = pd.merge(df, dfg, left_index=True, right_index=True, how='left')

    df.drop(['YEAR', 'MONTH', 'DAY'], axis=1, inplace=True)
    df.fillna(0.0, inplace=True)

    return df

# =================================================================================================================
def data_holiday():

    df = pd.read_csv(os.path.join(data_dir, 'HOLIDAY_20210117.csv'), \
                     engine='python', index_col='DT', parse_dates=True, encoding='utf-8')
    df.reset_index(inplace=True)
    df.drop(['IDX', 'NAME'], axis=1, inplace=True)
    df.sort_index(inplace=True)

    df['YEAR'] = df['DT'].map(lambda x: x.year)
    df['MONTH'] = df['DT'].map(lambda x: x.month)
    df['DAY'] = df['DT'].map(lambda x: x.day)
    df['HOLIDAY'] = 1.0

    df.drop_duplicates(['DT'], keep='first', inplace=True)
    df.set_index('DT', drop=True, inplace=True)

    return df

# =================================================================================================================
def prepare_dataset(df_thr):

    df_thr = data_thermaluse(df_thr)
    df_wth = data_weather()
    df_hol = data_holiday()

    # ---------------------------------------------------------------------------------------------------------
    df = pd.merge(df_thr, df_wth, left_index=True, right_index=True, how='left')

    df.reset_index(inplace=True)
    df_hol.reset_index(inplace=True, drop=True)
    df = pd.merge(left=df, right=df_hol, on=['YEAR', 'MONTH', 'DAY'], how='left')

    df['HOLIDAY'].fillna(0.0, inplace=True)
    df.set_index('DT', drop=True, inplace=True)
    df.index = df.index.astype('datetime64[ns]')

    df.reset_index(inplace=True)
    df['DAY_WEEK'] = df['DT'].apply(lambda x: 6 - x.weekday())
    df.set_index('DT', drop=True, inplace=True)

    df['WEEKEND'] = df['DAY_WEEK'].apply(lambda x: 0 if (x == 5 or x == 6) else 1)

    for dtid, dtval in df.iterrows():
        if dtval['HOLIDAY'] == 0.0:
            df.at[dtid, 'HOLIDAY'] = 5.0
        elif dtval['HOLIDAY'] == 1.0:
            try:
                df.at[(dtid - datetime.timedelta(days=1)), 'HOLIDAY'] = 3.0
            except:
                pass

    # ---------------------------------------------------------------------------------------------------------
    df.drop_duplicates(subset=['YEAR', 'MONTH', 'DAY', 'HOUR'], keep='first', inplace=True)
    dfidx = df.index

    df['count'] = 1

    df['gby_count_MDH'] = df.groupby(['MONTH', 'DAY_WEEK', 'HOUR'])['count'].cumcount()
    df['gby_count_MDH'].fillna(1, inplace=True)
    df['sum_gby_MDH'] = df.groupby(['MONTH', 'DAY_WEEK', 'HOUR'])['LOAD'].cumsum()
    df['mean_gby_MDH'] = df['sum_gby_MDH'] / df['gby_count_MDH']
    df.drop(['sum_gby_MDH'], axis=1, inplace=True)

    df['gby_count_MHH'] = df.groupby(['MONTH', 'HOLIDAY', 'HOUR'])['count'].cumcount()
    df['gby_count_MHH'].fillna(1, inplace=True)
    df['sum_gby_MHH'] = df.groupby(['MONTH', 'HOLIDAY', 'HOUR'])['LOAD'].cumsum()
    df['mean_gby_MHH'] = df['sum_gby_MHH'] / df['gby_count_MHH']
    df.drop(['sum_gby_MHH'], axis=1, inplace=True)

    df['gby_count_MWH'] = df.groupby(['MONTH', 'WEEKEND', 'HOUR'])['count'].cumcount()
    df['gby_count_MWH'].fillna(1, inplace=True)
    df['sum_gby_MWH'] = df.groupby(['MONTH', 'WEEKEND', 'HOUR'])['LOAD'].cumsum()
    df['mean_gby_MWH'] = df['sum_gby_MWH'] / df['gby_count_MWH']
    df.drop(['sum_gby_MWH'], axis=1, inplace=True)

    df['gby_YEAR'] = df['YEAR'].apply(lambda x: x + 1)
    dfg_MDH = df[['gby_YEAR', 'MONTH', 'HOUR', 'DAY_WEEK', 'mean_gby_MDH']]
    dfg_MDH.rename(columns={'gby_YEAR': 'YEAR'}, inplace=True)
    dfg_MHH = df[['gby_YEAR', 'MONTH', 'HOUR', 'HOLIDAY', 'mean_gby_MHH']]
    dfg_MHH.rename(columns={'gby_YEAR': 'YEAR'}, inplace=True)
    dfg_MWH = df[['gby_YEAR', 'MONTH', 'HOUR', 'WEEKEND', 'mean_gby_MWH']]
    dfg_MWH.rename(columns={'gby_YEAR': 'YEAR'}, inplace=True)
    df.drop(['count', 'gby_count_MDH', 'gby_count_MHH', 'gby_count_MWH', 'gby_YEAR', \
             'mean_gby_MDH', 'mean_gby_MHH', 'mean_gby_MWH'], axis=1, inplace=True)

    df = pd.merge(left=df, right=dfg_MDH, on=['YEAR', 'MONTH', 'DAY_WEEK', 'HOUR'], how='left',
                  suffixes=('', '_gby'))
    df.drop_duplicates(subset=['YEAR', 'MONTH', 'DAY', 'HOUR'], keep='first', inplace=True)
    df = pd.merge(left=df, right=dfg_MHH, on=['YEAR', 'MONTH', 'HOLIDAY', 'HOUR'], how='left',
                  suffixes=('', '_gby'))
    df.drop_duplicates(subset=['YEAR', 'MONTH', 'DAY', 'HOUR'], keep='first', inplace=True)
    df = pd.merge(left=df, right=dfg_MWH, on=['YEAR', 'MONTH', 'WEEKEND', 'HOUR'], how='left',
                  suffixes=('', '_gby'))
    df.drop_duplicates(subset=['YEAR', 'MONTH', 'DAY', 'HOUR'], keep='first', inplace=True)
    df.set_index(dfidx, drop=True, inplace=True)

    # ---------------------------------------------------------------------------------------------------------
    tmr_cols = ['MONTH', 'DAY', 'HOUR', 'HOLIDAY', 'DAY_WEEK', 'WEEKEND', \
                'TEMPERATURE', 'PRECIPITATION', 'WINDS_PEED', 'HUMIDITY', 'SNOWFALL', \
                'precipitation_cumsum', 'snowfall_cumsum', 'humidity_cumsum', \
                'mean_gby_MDH', 'mean_gby_MHH', 'mean_gby_MWH']

    for tc in tmr_cols:
        df[tc] = df[tc].shift(-25)
    df.drop(['YEAR'], axis=1, inplace=True)

    # ---------------------------------------------------------------------------------------------------------
    df.sort_index(inplace=True)
    df.fillna(0.0, inplace=True)
    df.replace([np.inf, -np.inf], 0.0, inplace=True)

    return df

# =================================================================================================================
def split_dataset(df, timesteps):

    df_train = df.loc['2012-01-01 00:00:00':'2019-01-01 00:00:00', :]
    df_val = df.loc['2019-01-01 00:00:00':'2020-01-01 00:00:00', :]
    df_test1 = df.loc['2020-01-01 00:00:00':'2020-02-01 00:00:00', :]
    df_test2 = df.loc['2020-04-01 00:00:00':'2020-09-01 00:00:00', :]
    df_test3 = df.loc['2020-11-01 00:00:00':'2020-12-26 00:00:00', :]

    # ---------------------------------------------------------------------------------------------------------
    scalerX = MinMaxScaler(feature_range=(0, 1))
    df_trainX = df_train.copy()
    scalerX.fit(df_trainX)
    scalerX_dir = os.path.join(model_dir, test_name + '-scaler.pickle')
    joblib.dump(scalerX, scalerX_dir)
    scalerX = joblib.load(scalerX_dir)
    scaled_features = scalerX.transform(df_trainX.values)
    df_trainX = pd.DataFrame(scaled_features, index=df_trainX.index, columns=df_trainX.columns)

    scalerY = MinMaxScaler(feature_range=(0, 1))
    df_trainY = pd.DataFrame()
    df_trainY['LOAD'] = df_train['LOAD']
    scalerY.fit(df_trainY)
    scalerY_dir = os.path.join(model_dir, test_name + '-scalerY.pickle')
    joblib.dump(scalerY, scalerY_dir)
    scalerY = joblib.load(scalerY_dir)
    scaled_features = scalerY.transform(df_trainY.values)
    df_trainY = pd.DataFrame(scaled_features, index=df_trainY.index, columns=df_trainY.columns)

    df_valX = df_val.copy()
    scaled_features = scalerX.transform(df_valX.values)
    df_valX = pd.DataFrame(scaled_features, index=df_valX.index, columns=df_valX.columns)
    df_valY = pd.DataFrame()
    df_valY['LOAD'] = df_val['LOAD']
    scaled_features = scalerY.transform(df_valY.values)
    df_valY = pd.DataFrame(scaled_features, index=df_valY.index, columns=df_valY.columns)

    df_testX1 = df_test1.copy()
    scaled_features = scalerX.transform(df_testX1.values)
    df_testX1 = pd.DataFrame(scaled_features, index=df_testX1.index, columns=df_testX1.columns)
    df_testY1 = pd.DataFrame()
    df_testY1['LOAD'] = df_test1['LOAD']
    df_testX2 = df_test2.copy()
    scaled_features = scalerX.transform(df_testX2.values)
    df_testX2 = pd.DataFrame(scaled_features, index=df_testX2.index, columns=df_testX2.columns)
    df_testY2 = pd.DataFrame()
    df_testY2['LOAD'] = df_test2['LOAD']
    df_testX3 = df_test3.copy()
    scaled_features = scalerX.transform(df_testX3.values)
    df_testX3 = pd.DataFrame(scaled_features, index=df_testX3.index, columns=df_testX3.columns)
    df_testY3 = pd.DataFrame()
    df_testY3['LOAD'] = df_test3['LOAD']

    # ---------------------------------------------------------------------------------------------------------
    dataX, dataY = [], []
    for k in range(0, len(df_trainY) - timesteps - 25):
        _x = df_trainX.iloc[k:k + timesteps].values.tolist()
        _y = df_trainY.iloc[k + timesteps + 25].values.tolist()
        dataX.append(_x)
        dataY.append(_y)
    trainX, trainY = np.array(dataX), np.array(dataY)

    dataX, dataY = [], []
    for k in range(0, len(df_valY) - timesteps - 25):
        _x = df_valX.iloc[k:k + timesteps].values.tolist()
        _y = df_valY.iloc[k + timesteps + 25].values.tolist()
        dataX.append(_x)
        dataY.append(_y)
    valX, valY = np.array(dataX), np.array(dataY)

    dataX, dataY, testDT = [], [], []
    for k in range(0, len(df_testY1) - timesteps - 25):
        _x = df_testX1.iloc[k:k + timesteps].values.tolist()
        _y = df_testY1.iloc[k + timesteps + 25].values.tolist()
        _dt = df_testY1.index[k + timesteps + 25]
        dataX.append(_x)
        dataY.append(_y)
        testDT.append(_dt)
    for k in range(0, len(df_testY2) - timesteps - 25):
        _x = df_testX2.iloc[k:k + timesteps].values.tolist()
        _y = df_testY2.iloc[k + timesteps + 25].values.tolist()
        _dt = df_testY2.index[k + timesteps + 25]
        dataX.append(_x)
        dataY.append(_y)
        testDT.append(_dt)
    for k in range(0, len(df_testY3) - timesteps - 25):
        _x = df_testX3.iloc[k:k + timesteps].values.tolist()
        _y = df_testY3.iloc[k + timesteps + 25].values.tolist()
        _dt = df_testY3.index[k + timesteps + 25]
        dataX.append(_x)
        dataY.append(_y)
        testDT.append(_dt)
    testX, testY = np.array(dataX), np.array(dataY)

    return trainX, trainY, valX, valY, testX, testY, testDT, scalerY

# =================================================================================================================
class attention_3d_block(Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, hidden_states):
        hidden_size = int(hidden_states.shape[2])
        hidden_states_t = Permute((2, 1), name='attention_input_t')(hidden_states)
        hidden_states_t = Reshape((hidden_size, timesteps), name='attention_input_reshape')(hidden_states_t)
        score_first_part = Dense(timesteps, use_bias=False, name='attention_score_vec')(hidden_states_t)
        score_first_part_t = Permute((2, 1), name='attention_score_vec_t')(score_first_part)
        h_t = Lambda(lambda x: x[:, :, -1], output_shape=(hidden_size, 1), name='last_hidden_state')(hidden_states_t)
        score = dot([score_first_part_t, h_t], [2, 1], name='attention_score')
        attention_weights = Activation('softmax', name='attention_weight')(score)
        context_vector = dot([hidden_states_t, attention_weights], [2, 1], name='context_vector')
        context_vector = Reshape((hidden_size,))(context_vector)
        h_t = Reshape((hidden_size,))(h_t)
        pre_activation = concatenate([context_vector, h_t], name='attention_output')
        attention_vector = Dense(hidden_size, use_bias=False, activation='tanh', name='attention_vector')(pre_activation)
        return attention_vector

# =================================================================================================================

def create_model(learning_rate, filter_val, kernel_val, num_lstm_nodes, activation):

    inputs = Input(shape=(timesteps, data_dim), name='inputs')

    conv_out1 = Conv1D(filters=filter_val, kernel_size=kernel_val, strides=1, activation='relu',
                       input_shape=(timesteps, data_dim), padding='same', name='Conv1D1')(inputs)
    conv_out2 = Conv1D(filters=num_lstm_nodes, kernel_size=kernel_val, strides=1, activation='relu',
                       input_shape=(timesteps, data_dim), padding='same', name='Conv1D2')(conv_out1)
    conv_out3 = Conv1D(filters=int(num_lstm_nodes/2), kernel_size=kernel_val, strides=1, activation='relu',
                       input_shape=(timesteps, data_dim), padding='same', name='Conv1D3')(conv_out2)
    bn_output1 = BatchNormalization(name='BatchNormalization1')(conv_out3)

    lstm_out1 = LSTM(filter_val, return_sequences=True, name='LSTM1', activation=activation, \
                     input_shape=(timesteps, data_dim))(inputs)
    lstm_out2 = LSTM(num_lstm_nodes, return_sequences=True, name='LSTM2', activation=activation)(lstm_out1)
    lstm_out3 = LSTM(int(num_lstm_nodes/2), return_sequences=True, name='LSTM3', activation=activation)(lstm_out2)
    bn_output2 = BatchNormalization(name='BatchNormalization2')(lstm_out3)

    concat_out = Concatenate()([bn_output1, bn_output2])
    bn_output3 = BatchNormalization(name='BatchNormalization3')(concat_out)
    attention_vec = attention_3d_block(name='attention_weight')(bn_output3)
    bn_output4 = BatchNormalization(name='BatchNormalization4')(attention_vec)
    att_pred2 = Dense(16, activation='linear', name='Dense2')(bn_output4)
    att_pred = Dense(1, activation='linear', name='Dense')(att_pred2)
    model = Model(inputs=[inputs], outputs=att_pred)

    opt = optimizers.Adam(lr=learning_rate, decay=0.01)
    model.compile(optimizer=opt, loss='mae', metrics=['mae'])

    return model

# =================================================================================================================
@use_named_args(dimensions=dimensions)
def fitness(learning_rate, filter_val, kernel_val, num_lstm_nodes, activation, batch_size):

    K.clear_session()

    parallel_model = create_model(learning_rate, filter_val, kernel_val, num_lstm_nodes, activation)
    checkpoint_dir = os.path.join(model_dir, 'cp-' + test_name + '.h5')
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir, verbose=2, save_weights_only=False, \
                                                     monitor='val_loss', mode='min', period=5, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, mode='min', patience=20)
    blackbox = parallel_model.fit(x=X_train, y=y_train, epochs=300, batch_size=batch_size,
                                  validation_data=(valX, valY), callbacks=[cp_callback, early_stopping], verbose=2)
    val_loss = blackbox.history['val_loss'][-1]
    print("MAE: {}".format(val_loss))

    K.clear_session()

    return val_loss

# =================================================================================================================
def result_performance(df):

    df['ACTUAL'] = round(df['ACTUAL'].astype(float), 3)
    df['PREDICT'] = round(df['PREDICT'].astype(float), 3)

    mae = metrics.mean_absolute_error(df['ACTUAL'], df['PREDICT'])
    mse = metrics.mean_squared_error(df['ACTUAL'], df['PREDICT'])
    rmse = math.sqrt(mse)
    print('mae :: ', mae)
    print('mse :: ', mse)
    print('rmse :: ', rmse)

    # ---------------------------------------------------------------------------------------------------------
    f, ax = plt.subplots(figsize=(40, 5))
    plt.plot(df['ACTUAL'], label='Actual', color='black')
    plt.plot(df['PREDICT'], label='Predicted', color='blue')
    plt.legend()
    fig_path = os.path.join(image_dir, test_name + '.png')
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()

    return df

# =================================================================================================================
def main():

    dff = pd.read_csv(os.path.join(data_dir, 'DH-20110101-20210116.csv'), \
                      engine='python', index_col='DT', parse_dates=True, encoding='utf-8')
    global data_dim, X_train, y_train, valX, valY
    df = prepare_dataset(dff)
    df.dropna(inplace=True)
    data_dim = df.shape[1]
    X_train, y_train, valX, valY, testX, testY, testDT, scalerY = split_dataset(df, timesteps)

    # ---------------------------------------------------------------------------------------------------------
    K.clear_session()
    gp_result = gp_minimize(func=fitness, dimensions=dimensions, n_calls=100, x0=default_parameters,
                            acq_func='gp_hedge', base_estimator='GP')

    print("gp_result gp_result gp_result")
    print("gp_result : ", gp_result)
    print("best MAE was " + str(round(gp_result.fun, 2)))
    print('gp_result.x ::: ', gp_result.x)
    print('gp_result.func_vals ::: ', gp_result.func_vals)

    print('retrain our best model architecture')
    K.clear_session()

    checkpoint_dir = os.path.join(model_dir, 'cp-' + test_name + '.h5')
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir, verbose=2, save_weights_only=False, \
                                                     monitor='val_loss', mode='min', period=5, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, mode='min', patience=20)
    gp_model = create_model(gp_result.x[0], gp_result.x[1], gp_result.x[2], gp_result.x[3], gp_result.x[4])
    gp_model.fit(x=X_train, y=y_train, epochs=300, batch_size=gp_result.x[5],
                 validation_data=(valX, valY), callbacks=[cp_callback, early_stopping], verbose=2)
    gp_model.save(os.path.join(model_dir, test_name + '.h5'))

    # ---------------------------------------------------------------------------------------------------------
    K.clear_session()
    print('Check Point Model ---------------------------------')
    gp_model = load_model(checkpoint_dir)
    predY = gp_model.predict(testX)
    predY = scalerY.inverse_transform(predY.reshape(-1, 1))
    predY = predY.reshape(1, -1)[0]

    df_rst = pd.DataFrame(columns=['DT', 'ACTUAL', 'PREDICT'])
    for rst_idx in range(0, len(testDT)):
        df_rst.loc[len(df_rst)] = [testDT[rst_idx], testY[rst_idx][0], predY[rst_idx]]
    result_performance(df_rst)

    K.clear_session()
    print('Model ---------------------------------')
    gp_model = load_model(os.path.join(model_dir, test_name + '.h5'))
    predY = gp_model.predict(testX)
    predY = scalerY.inverse_transform(predY.reshape(-1, 1))
    predY = predY.reshape(1, -1)[0]

    df_rst = pd.DataFrame(columns=['DT', 'ACTUAL', 'PREDICT'])
    for rst_idx in range(0, len(testDT)):
        df_rst.loc[len(df_rst)] = [testDT[rst_idx], testY[rst_idx][0], predY[rst_idx]]
    result_performance(df_rst)

    K.clear_session()

    return 0

# =================================================================================================================
if __name__ == "__main__":
    main()

