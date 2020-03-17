from pyspark.sql.types import StructType, StructField, DateType, IntegerType

from demo_2 import Configs
import gc


useless_cols = ['vessel_code', 'foTotVolume', 'scavAirInLetPress', 'scavengeAirPressure', 'scavAirFireDetTempNo11',
 'scavAirFireDetTempNo12',  'coolerCWinTemp', 'cylLoTemp', 'hfoViscocityHighLow', 'hpsBearingTemp', 'cylExhGasOutTempNo11',
 'cylExhGasOutTempNo12', 'cylJCFWOutTempNo11', 'cylJCFWOutTempNo12', 'cylPistonCOOutTempNo11', 'cylPistonCOOutTempNo12',
 'tcExhGasInTempNo4', 'tcExhGasOutTempNo4', 'tcLOInLETPressNo4', 'tcLOOutLETTempNo4',
 'tcRPMNo4', 'orderRPMBridgeLeverer', 'scavengeAirPressure', 'coolingWOutLETTempNo4',
 'foConsumption', 'foID','foflow']

def load_vessel_data_of_specific_vessel(vessel_id=None, db_table='_vds.csv', sc=None):

    schema = StructType([
        StructField("datetime", DateType()),
        StructField("stw", IntegerType()),
        StructField("speed_overground", IntegerType()),
        ])

    # train_vds = sc.read.csv(Configs.INPUT_PATH +'tail_'+ str(vessel_id) + '_vds.csv', nullValue=0, header=True, inferSchema=True)
    train_vds = sc.read.csv(Configs.INPUT_PATH + str(vessel_id) + db_table, header=True, inferSchema=True)
    train_vds = train_vds[train_vds['datetime'] > '2015-10-31 23:00:00']

    return train_vds


def load_main_engine_data_of_specific_vessel(vessel_id=None,  sc=None, db_table='_mes.csv'):

    schema = StructType([
        StructField("datetime", DateType()),
        StructField("airCoolerCWInLETPress", IntegerType()),
        StructField("scavAirFireDetTempNo1", IntegerType()),
        StructField("scavAirFireDetTempNo2", IntegerType()),
        StructField("scavAirFireDetTempNo3", IntegerType()),
        StructField("scavAirFireDetTempNo4", IntegerType()),
        StructField("scavAirFireDetTempNo5", IntegerType()),
        StructField("scavAirFireDetTempNo6", IntegerType()),
        StructField("scavAirFireDetTempNo7", IntegerType()),
        StructField("scavAirFireDetTempNo8", IntegerType()),
        StructField("scavAirFireDetTempNo9", IntegerType()),
        StructField("scavAirFireDetTempNo10", IntegerType()),
        StructField("cfWInPress", IntegerType()),
        StructField("controlAirPress", IntegerType()),
        StructField("exhVVSpringAirInPress", IntegerType()),
        StructField("foFlow", IntegerType()),
        StructField("foInPress", IntegerType()),
        StructField("foInTemp", IntegerType()),
        StructField("jcfWInTempLow", IntegerType()),
        StructField("cylExhGasOutTempNo1", IntegerType()),
        StructField("cylExhGasOutTempNo2", IntegerType()),
        StructField("cylExhGasOutTempNo3", IntegerType()),
        StructField("cylExhGasOutTempNo4", IntegerType()),
        StructField("cylExhGasOutTempNo5", IntegerType()),
        StructField("cylExhGasOutTempNo6", IntegerType()),
        StructField("cylExhGasOutTempNo7", IntegerType()),
        StructField("cylExhGasOutTempNo8", IntegerType()),
        StructField("cylExhGasOutTempNo9", IntegerType()),
        StructField("cylExhGasOutTempNo10", IntegerType()),
        StructField("cylJCFWOutTempNo1", IntegerType()),
        StructField("cylJCFWOutTempNo2", IntegerType()),
        StructField("cylJCFWOutTempNo3", IntegerType()),
        StructField("cylJCFWOutTempNo4", IntegerType()),
        StructField("cylJCFWOutTempNo5", IntegerType()),
        StructField("cylJCFWOutTempNo6", IntegerType()),
        StructField("cylJCFWOutTempNo7", IntegerType()),
        StructField("cylJCFWOutTempNo8", IntegerType()),
        StructField("cylJCFWOutTempNo9", IntegerType()),
        StructField("cylJCFWOutTempNo10", IntegerType()),
        StructField("cylPistonCOOutTempNo1", IntegerType()),
        StructField("cylPistonCOOutTempNo2", IntegerType()),
        StructField("cylPistonCOOutTempNo3", IntegerType()),
        StructField("cylPistonCOOutTempNo4", IntegerType()),
        StructField("cylPistonCOOutTempNo5", IntegerType()),
        StructField("cylPistonCOOutTempNo6", IntegerType()),
        StructField("cylPistonCOOutTempNo7", IntegerType()),
        StructField("cylPistonCOOutTempNo8", IntegerType()),
        StructField("cylPistonCOOutTempNo9", IntegerType()),
        StructField("cylPistonCOOutTempNo10", IntegerType()),
        StructField("tcExhGasInTempNo1", IntegerType()),
        StructField("tcExhGasInTempNo2", IntegerType()),
        StructField("tcExhGasInTempNo3", IntegerType()),
        StructField("tcExhGasOutTempNo1", IntegerType()),
        StructField("tcExhGasOutTempNo2", IntegerType()),
        StructField("tcExhGasOutTempNo3", IntegerType()),
        StructField("tcLOInLETPressNo1", IntegerType()),
        StructField("tcLOInLETPressNo2", IntegerType()),
        StructField("tcLOInLETPressNo3", IntegerType()),
        StructField("tcLOOutLETTempNo1", IntegerType()),
        StructField("tcLOOutLETTempNo2", IntegerType()),
        StructField("tcLOOutLETTempNo3", IntegerType()),
        StructField("tcRPMNo1", IntegerType()),
        StructField("tcRPMNo2", IntegerType()),
        StructField("tcRPMNo3", IntegerType()),
        StructField("rpm", IntegerType()),
        StructField("scavAirReceiverTemp", IntegerType()),
        StructField("startAirPress", IntegerType()),
        StructField("thrustPadTemp", IntegerType()),
        StructField("mainLOInLetPress", IntegerType()),
        StructField("mainLOInTemp", IntegerType()),
        StructField("foTemperature", IntegerType()),
        StructField("power", IntegerType()),
        StructField("torque", IntegerType()),
        StructField("coolingWOutLETTempNo1", IntegerType()),
        StructField("coolingWOutLETTempNo2", IntegerType()),
        StructField("coolingWOutLETTempNo3", IntegerType()),
        StructField("foVolConsumption", IntegerType()),

    ])

    # train = sc.read.csv(Configs.INPUT_PATH + 'tail_'+ str(vessel_id) + '_mes.csv', nullValue=0, header=True, inferSchema=True)

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

        print(train.printSchema())
        print(train.show(n=10))


        return train

    if model == 'svm' or model == 'lstm':
        train_vds = load_vessel_data_of_specific_vessel(7, sc=spark)
        train = load_main_engine_data_of_specific_vessel(7,sc=spark)
        train_vds_2 = load_vessel_data_of_specific_vessel(8,sc=spark)
        train_2 = load_main_engine_data_of_specific_vessel(8,sc=spark)

        train_vds = train_vds.select(['datetime', 'stw', 'speed_overground'])
        train_vds_2 = train_vds_2.select(['datetime', 'stw', 'speed_overground'])
        print('length :' + str(len(train.columns)))
        train = train.join(train_vds, ['datetime'])
        train_2 = train_2.join(train_vds_2, ['datetime'])
        print('length :' + str(len(train.columns)))
        print('Test Rows : ' + str(train.count()))
        train = train.union(train_2)
        print('length :' + str(len(train.columns)))
        print('Test Rows : ' + str(train.count()))


        del train_2,train_vds,train_vds_2
        gc.collect()

        train = train.drop(*useless_cols)
        print(train.printSchema())
        print(train.show(n=2))

        return train


def load_test_data(spark):

    test_vds= load_vessel_data_of_specific_vessel(5, sc=spark)
    test = load_main_engine_data_of_specific_vessel(5, sc= spark)

    test_vds = test_vds.select(['datetime','stw','speed_overground'])
    print('length :' + str(len(test.columns)))

    print(test.show(5))
    print(test_vds.show(5))

    test = test.join(test_vds,['datetime'])
    print(test.printSchema())

    print('length :' + str(len(test.columns)))
    del test_vds
    gc.collect()

    test = test.drop(*useless_cols)
    print('length after drop :' + str(len(test.columns)))

    print(test.printSchema())
    print(test.show(n=2))
    return test

# def send_anomalies_to_db(dataframe):
#     dataframe.to_sql(dataframe, 'table_name', connection)
