import xgboost as xgb
from utils import create_splits
import torch
import gc


def train_xgb_model(model_name, x_train, x_dev, y_train, y_dev, params,
                    debug=False):
    num_boost_round = 1 if debug else 3000

    dtrain = xgb.DMatrix(data=x_train, label=y_train)
    dvalid = xgb.DMatrix(data=x_dev, label=y_dev)
    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    bst = xgb.train(params, dtrain=dtrain,
                    num_boost_round=num_boost_round, evals=watchlist,
                    early_stopping_rounds=500, maximize=False,
                    verbose_eval=2000)
    bst.save_model(model_name + ".txt")
    return bst.best_score


def predict_xgb_model(model_name, x_test):
    dtest = xgb.DMatrix(data=x_test)
    model = xgb.Booster()
    model.load_model(model_name + ".txt")
    return model.predict(dtest)


def train_models(X, Y, D, F, debug=False):
    P = []
    params = {'lambda': 2.891676910822147, 'alpha': 0.0015541716316951784, 'colsample_bytree': 0.5, 'subsample': 0.6,
              'learning_rate': 0.012, 'random_state': 20230618, 'min_child_weight': 5, 'eta': 5.8514190695165885e-06,
              'gamma': 0.37785760883613917, 'grow_policy': 'lossguide', 'max_depth': 3}
    P.append(params)

    params = {'lambda': 2.8103719674545617, 'alpha': 0.03499536170701692, 'colsample_bytree': 0.6, 'subsample': 0.8,
              'learning_rate': 0.014, 'random_state': 20230617, 'min_child_weight': 1, 'eta': 6.4241191526913975e-06,
              'gamma': 6.14332027852065e-06, 'grow_policy': 'lossguide'}
    params['max_depth'] = 5
    P.append(params)

    params = {'lambda': 0.0349979156441287, 'alpha': 0.019213865079303578, 'colsample_bytree': 0.9, 'subsample': 0.8,
              'learning_rate': 0.01, 'random_state': 20230616, 'min_child_weight': 7, 'eta': 2.302540548857049e-05,
              'gamma': 0.00012413661649583299, 'grow_policy': 'lossguide', 'max_depth': 4}
    P.append(params)

    S = [
        [11111, 22222, 33333, 44444, 55555, 12345],
        [66666, 77777, 88888, 99999, 00000, 10101],
        [12312, 23423, 34535, 45645, 56756, 67867]
    ]

    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        print('no gpu detected, expect much slower training times')

    assert len(S) == len(P)
    scores = []
    i = 0
    for params, seeds in zip(P, S):
        params['objective'] = 'binary:logistic'
        if cuda_available:
            params['sampling_method'] = 'gradient_based'
            params['tree_method'] = 'gpu_hist'

        for s in seeds:
            x_train, x_dev, y_train, y_dev, _, _ = create_splits(X, Y, D, F, seed=s)
            gc.collect()

            model_path = f'xgb-prob-model{i}'
            i += 1

            s = train_xgb_model(model_path, x_train, x_dev, y_train, y_dev,
                                params=params, debug=debug)
            scores.append(s)

    return sum(scores) / len(scores)
