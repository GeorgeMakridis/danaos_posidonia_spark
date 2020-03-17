import demo_2.preprocess as preprocess
import demo_2.load_data as load_data
from demo_2.DetectorXGB import DetectorXGB
import pandas as pd


class DetectorLevel_1(object):
    

    def __init__(self,param, sc):
        self.param =param
        self.test_ini = None
        self.train_ini = None
        self.sc=sc


    def run_xgb_prediction(self, message):
        print(message)
        if self.test_ini==None:
            self.test_ini=load_data.load_test_data(self.sc)
        # test = load_data.load_test_data()
        test = preprocess.preprocess_test(self.test_ini, 'xgb')
        xgb_detector = DetectorXGB(test, self.sc)
        xgb_detector.predict(test)


    def save_prediction_result(self, message):
        print(message)
        pd_results = pd.read_csv('results_1_level.csv', index_col='datetime', parse_dates=['datetime'])
        pd_results['result_anomaly'] = pd_results.mean(axis=1)
        load_data.send_anomalies_to_db(pd_results)


    def run_xgb_training(self, message):
        print(message)
        if self.train_ini == None:
            self.train_ini = load_data.load_train_data('xgb', spark=self.sc)
        # train = load_data.load_train_data('xgb')
        train = preprocess.preprocess_train(self.train_ini, 'xgb')
        xgb_detector = DetectorXGB(train, self.sc)
        xgb_detector.fit(train)



