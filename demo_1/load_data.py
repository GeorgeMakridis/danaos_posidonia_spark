import gc

from demo_1 import Configs

useless_cols = ['vessel_code', 'foTotVolume', 'scavAirInLetPress', 'scavengeAirPressure', 'scavAirFireDetTempNo11',
 'scavAirFireDetTempNo12',  'coolerCWinTemp', 'cylLoTemp', 'hfoViscocityHighLow', 'hpsBearingTemp', 'cylExhGasOutTempNo11',
 'cylExhGasOutTempNo12', 'cylJCFWOutTempNo11', 'cylJCFWOutTempNo12', 'cylPistonCOOutTempNo11', 'cylPistonCOOutTempNo12',
 'tcExhGasInTempNo4', 'tcExhGasOutTempNo4', 'tcLOInLETPressNo4', 'tcLOOutLETTempNo4',
 'tcRPMNo4', 'orderRPMBridgeLeverer', 'scavengeAirPressure', 'coolingWOutLETTempNo4',
 'foConsumption', 'foID']


def load_vessel_data_of_specific_vessel(vessel_id=None, db_table='_vds.csv', sc=None):
    train_vds = sc.read.csv(Configs.INPUT_PATH + str(vessel_id) + db_table, header=True, inferSchema=True)
    train_vds = train_vds[train_vds['datetime'] > '2015-10-31 23:00:00']
    return train_vds


def load_main_engine_data_of_specific_vessel(vessel_id=None,  sc=None, db_table='_mes.csv'):
    train = sc.read.csv(Configs.INPUT_PATH + str(vessel_id) + db_table, header=True, inferSchema=True)
    train = train.dropDuplicates(['datetime'])
    train = train[train['datetime'] > '2015-10-31 23:00:00']
    return train


def load_train_data(model=None, spark=None):
    if model == 'xgb':

        train_vds = load_vessel_data_of_specific_vessel(7, sc=spark)
        train = load_main_engine_data_of_specific_vessel(7, sc=spark)

        train_vds = train_vds.select(['datetime', 'stw', 'speed_overground'])
        print('length :' + str(len(train.columns)))
        train = train.join(train_vds, ['datetime'])
        print('length :' + str(len(train.columns)))
        print('Test Rows : ' + str(train.count()))

        train = train.drop(*useless_cols)


        train = train.toPandas()

        train.set_index('datetime',inplace=True)
        print(train.head())
        return train


def load_test_data(spark):
    test_vds= load_vessel_data_of_specific_vessel(5, sc=spark)
    test = load_main_engine_data_of_specific_vessel(5, sc= spark)

    test_vds = test_vds.select(['datetime','stw','speed_overground'])

    print('length :' + str(len(test.columns)))

    test = test.join(test_vds,['datetime'])

    print('length :' + str(len(test.columns)))
    del test_vds
    gc.collect()

    test = test.drop(*useless_cols)
    print('length after drop :' + str(len(test.columns)))
    test = test.toPandas()
    test.set_index('datetime', inplace=True)
    return test

# def send_anomalies_to_db(dataframe):
#     dataframe.to_sql(dataframe, 'table_name', connection)
