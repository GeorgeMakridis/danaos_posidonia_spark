import pandas as pd
from  demo_1 import preprocess
from demo_1.DetectorXGB import DetectorXGB

from demo_1 import load_data


class DetectorLevel_1(object):
    
    def __init__(self,param, sc):
        self.param =param
        self.test_ini = None
        self.train_ini = None
        self.sc=sc


    def run_xgb_prediction(self, message):
        print(message)
        if self.test_ini is None:
            self.test_ini= load_data.load_test_data(self.sc)
        test = preprocess.preprocess_test(self.test_ini, 'xgb')
        xgb_detector = DetectorXGB(test)
        xgb_detector.predict(test)

    def save_prediction_result(self, message):
        print(message)
        pd_results = pd.read_csv('results_1_level.csv', index_col='datetime', parse_dates=['datetime'])
        pd_results['result_anomaly'] = pd_results.mean(axis=1)
        load_data.send_anomalies_to_db(pd_results)


    def run_xgb_training(self, message):
        print(message)
        if self.train_ini is None:
            self.train_ini = load_data.load_train_data('xgb', spark=self.sc)
        train = preprocess.preprocess_train(self.train_ini, 'xgb')
        xgb_detector = DetectorXGB(train)
        xgb_detector.fit(train)


