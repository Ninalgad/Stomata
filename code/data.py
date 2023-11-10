from datetime import timedelta
import numpy as np
from scipy.stats import kurtosis, skew
import warnings
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
import os
import pandas as pd
import pickle

FEAT = ['pump_speed', 'pump_current', 'motor_temp', 'pump_temp', 'fluid_press',
        'fluid_temp', 'motor_trq', 'motor_neutral_v']
META_CONT_FEAT = ['pump_size', 'motor_power',
                  'location_coordinate_1', 'location_coordinate_2']


def features_above_percentile(x, p):
    idx = x > np.percentile(x, p)
    x = x[idx]
    return [np.log1p(x.sum()),  # log of "stop-loss treaty" above a percentile amount
            x.mean(),  # "expected shortfall"
            np.log1p(idx.sum()), idx.mean()
            ]


def describe_values(x):
    x_not_nan = x[np.logical_not(np.isnan(x))]
    n_not_nan = len(x_not_nan)
    if n_not_nan == 0:
        return list(-np.ones(42, 'uint8'))
    n = len(x)
    nan_prop = 10 * n_not_nan / n

    loc_max = np.argmax(x) / n
    loc_min = np.argmin(x) / n
    feat = [
        loc_max, loc_min, 10 * (loc_max - loc_min),
        nan_prop, np.log1p(n), np.log1p(n_not_nan),
        np.max(x_not_nan), np.min(x_not_nan), np.mean(x_not_nan),
        np.log1p(np.sum(x_not_nan)),
        np.std(x_not_nan), skew(x_not_nan), kurtosis(x_not_nan)
    ]

    feat += [np.percentile(x_not_nan, 95), np.percentile(x_not_nan, 66.66),
             np.percentile(x_not_nan, 80), np.percentile(x_not_nan, 90),
             np.percentile(x_not_nan, 99)]

    for p in [25, 50, 66.66, 80, 90, 95]:
        feat += features_above_percentile(x_not_nan, p)

    assert len(feat) == 42, len(feat)
    return feat


def create_sub_features(df):
    features = []
    for f in FEAT:
        x = np.log1p(df[f].values)
        dx = x[1:] - x[:-1]
        dx2 = x[2:] - x[:-2]

        features += describe_values(x)
        features += describe_values(dx)
        features += describe_values(dx2)

    x = df[FEAT].values
    x = np.array(features, 'float32')
    return x


def create_workload_features(df, dt):
    latest_time = dt.iloc[-1]

    times = [{'days': 1}, {'days': 7}]
    x_full = create_sub_features(df)
    features = [
        [np.log1p((dt.iloc[-1] - dt.iloc[0]).days)],  # log-age
        x_full
    ]
    for t in times:
        sub_df = df[dt > (latest_time - timedelta(**t))]
        features.append(create_sub_features(sub_df))
    return np.nan_to_num(np.concatenate(features), nan=-1, neginf=-1, posinf=-1)


def create_meta_features(df_meta, motor_ohe, pump_ohe, loc_ohe):
    m = motor_ohe.transform([[df_meta.motor_type]])[0]
    p = pump_ohe.transform([[df_meta.pump_type]])[0]
    l = loc_ohe.transform([[df_meta.location]])[0]
    c = df_meta[META_CONT_FEAT].values.astype('float32')
    return np.concatenate([m, p, l, c], dtype='float32')


def create_features(df, dt, meta, motor_ohe, pump_ohe, loc_ohe):
    meta_feat = create_meta_features(meta, motor_ohe, pump_ohe, loc_ohe)
    wl_feat = create_workload_features(df, dt)
    return np.concatenate([meta_feat, wl_feat], dtype='float32')


def days2label(num_days):
    days = [3, 7, 14, 30, 45, 60, 120]
    return np.array([num_days < d for d in days], 'float32')


def create_or_load_ohe(file_name, data, force_fit_ohe=False):
    file_name = file_name + '.pkl'
    if force_fit_ohe or (not os.path.isfile(file_name)):
        ohe_obj = OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=10, sparse_output=False)
        ohe_obj.fit(data)
        with open(file_name, 'wb') as f:
            pickle.dump(ohe_obj, f)
    else:
        with open(file_name, 'rb') as f:
            ohe_obj = pickle.load(f)
    return ohe_obj


def _create_train_dataset(meta_data, data_dir, motor_ohe, pump_ohe, loc_ohe,
                          min_timesteps=500, eps=1):
    X, Y, F, D = [], [], [], []
    days_until_failure = [0, 1, 3, 4, 7, 10, 14, 19, 30, 38, 45, 50,
                          60, 80, 100, 120, 150, 200, 360]

    for fnum, meta in tqdm(meta_data.iterrows(), total=len(meta_data)):
        df = pd.read_csv(data_dir + f'/operations-data/{fnum}.csv')
        dt = pd.to_datetime(df.date, format="%m/%d/%Y %H:%M:%S %p")
        if meta.failed_date == -1:
            latest_time = dt.iloc[-1]
        else:
            latest_time = pd.to_datetime(meta.failed_date, format="%Y-%m-%d")

        if meta.status == 'Removed':
            x = create_features(df, dt, meta, motor_ohe, pump_ohe, loc_ohe)
            X.append(x)
            y = days2label(0)
            Y.append(y)
            D.append(0)
            F.append(fnum)
            for i, d in enumerate(days_until_failure):
                idx = dt <= latest_time - timedelta(days=d + eps)
                if idx.sum() == 0:
                    continue
                elif (idx.sum() < min_timesteps) and (i > 0):
                    continue
                x = create_features(df[idx], dt[idx], meta, motor_ohe, pump_ohe, loc_ohe)
                X.append(x)
                y = days2label(d)
                Y.append(y)
                D.append(d + eps)
                F.append(fnum)

        else:
            idx = dt <= latest_time - timedelta(days=120 + eps)
            if idx.sum() < min_timesteps:
                continue
            x = create_features(df[idx], dt[idx], meta, motor_ohe, pump_ohe, loc_ohe)
            X.append(x)
            y = days2label(120)
            Y.append(y)
            D.append(120 + eps)
            F.append(fnum)

    X = np.array(X, 'float32')
    Y = np.array(Y, 'uint8')
    D = np.log1p(np.array(D, 'float32'))
    F = np.array(F)
    return X, Y, D, F


def process_training_data(data_dir, fit_ohe=False, debug=False):
    df_meta = pd.read_csv(data_dir + '/equipment_metadata.csv', index_col='failure_number')
    df_meta = df_meta.fillna(-1)

    motor_ohe = create_or_load_ohe('motor_ohe', df_meta.motor_type.values[:, None], fit_ohe)
    pump_ohe = create_or_load_ohe('pump_ohe', df_meta.pump_type.values[:, None], fit_ohe)
    loc_ohe = create_or_load_ohe('loc_ohe', df_meta.location.values[:, None], fit_ohe)

    if debug:
        df_meta = df_meta.iloc[:10]

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
        warnings.filterwarnings('ignore', 'Mean of empty slice')
        warnings.filterwarnings('ignore', 'Degrees of freedom <= 0 for slice.')
        warnings.filterwarnings('ignore', 'invalid value encountered in log1p')
        warnings.filterwarnings('ignore', 'divide by zero encountered in log1p')
        warnings.filterwarnings('ignore', 'Precision loss occurred')
        warnings.filterwarnings('ignore', 'invalid value encountered in double_scalars')

        return _create_train_dataset(df_meta, data_dir, motor_ohe, pump_ohe, loc_ohe)


def _create_test_dataset(meta_data, data_dir, motor_ohe, pump_ohe, loc_ohe):
    X, F = [], []
    for fnum, meta in tqdm(meta_data.iterrows(), total=len(meta_data)):
        df = pd.read_csv(data_dir + f'/operations-data/{fnum}.csv')
        dt = pd.to_datetime(df.date, format="%m/%d/%Y %H:%M:%S %p")

        x = create_features(df, dt, meta, motor_ohe, pump_ohe, loc_ohe)
        X.append(x)
        F.append(fnum)

    return np.array(X, 'float32'), F


def process_test_data(data_dir, debug=False):
    df_meta = pd.read_csv(data_dir + '/equipment_metadata.csv', index_col='failure_number')
    df_meta = df_meta.fillna(-1)

    motor_ohe = create_or_load_ohe('motor_ohe', df_meta.motor_type.values[:, None])
    pump_ohe = create_or_load_ohe('pump_ohe', df_meta.pump_type.values[:, None])
    loc_ohe = create_or_load_ohe('loc_ohe', df_meta.location.values[:, None])

    if debug:
        df_meta = df_meta.iloc[:10]

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
        warnings.filterwarnings('ignore', 'Mean of empty slice')
        warnings.filterwarnings('ignore', 'Degrees of freedom <= 0 for slice.')
        warnings.filterwarnings('ignore', 'invalid value encountered in log1p')
        warnings.filterwarnings('ignore', 'divide by zero encountered in log1p')
        warnings.filterwarnings('ignore', 'Precision loss occurred')
        warnings.filterwarnings('ignore', 'invalid value encountered in double_scalars')

        return _create_test_dataset(df_meta, data_dir, motor_ohe, pump_ohe, loc_ohe)
