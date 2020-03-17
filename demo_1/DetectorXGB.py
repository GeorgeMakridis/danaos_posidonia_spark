import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import xgboost as xgb
import os


class DetectorXGB:
    def __init__(self, data):
        self.data = data

    def fit(self, train):
        y = train['label'].fillna(0)
        x = train.drop(['label'], axis=1, inplace=False)

        scale_pos = (len(y) - sum(y.values)) / sum(y.values)

        params = {"objective": "binary:logistic",
                  "eta": 0.05,
                  "max_depth": 4,
                  "subsample": 1,
                  "colsample_bytree": 1,
                  "silent": 1,
                  "scale_pos_weight": scale_pos,
                  'lambda': 1
                  }
        num_boost_round = 1000

        print("Train a XGBoost model")
        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.5, random_state=13, stratify=y, )
        dtrain = xgb.DMatrix(x_train, y_train)
        dvalid = xgb.DMatrix(x_valid, y_valid)

        watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
        gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=10, verbose_eval=True)

        # print("Saving the XGB model")
        # pickle.dump(gbm, open("1_leve_xgb_model_1Hour.pickle.dat", "wb"))

    def predict(self, test):
        import pickle
        test_xgb = test

        x = test_xgb
        df = test_xgb

        # load model
        loaded_model_gbm = pickle.load(open('1_leve_xgb_model_1Hour.pickle.dat', 'rb'))

        print(df.shape)
        print(test_xgb.shape)
        print(len(loaded_model_gbm.predict(xgb.DMatrix(x))))
        df['yhat'] = loaded_model_gbm.predict(xgb.DMatrix(x))

        df['20 Day MA'] = df['yhat'].rolling(window=20).mean()
        df['20 Day STD'] = df['yhat'].rolling(window=20).std()
        #
        df['Upper Band 20'] = df['20 Day MA'] + (df['20 Day STD'] * 4)

        df['anomaly'] = 0
        df['anomaly'].loc[df['yhat'] > df['Upper Band 20']] = 1

        root_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(root_dir, 'results_1_level.csv')
        exists = os.path.isfile(file_path)
        if exists:
            pd_results = pd.read_csv('results_1_level.csv', index_col='datetime', parse_dates=['datetime'])
        else:
            pd_results = pd.DataFrame()
        pd_results['xgb'] = df['anomaly']

        pd_results.to_csv('results_1_level.csv', index=True)

