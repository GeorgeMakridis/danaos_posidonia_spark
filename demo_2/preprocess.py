from datetime import timedelta

import sys

from pyspark.ml.feature import StandardScaler, Normalizer, VectorAssembler, PCA
from pyspark.sql import Window
from pyspark.sql.types import DoubleType
from sklearn.decomposition import FastICA
from sklearn.preprocessing import normalize
from sklearn.preprocessing import scale
from joblib import dump, load

from level_1_helper_functions import clip_data, get_mean_of_cyl_values, clip, standardize_train_test_data
import Configs
import pandas as pd
import gc
from pyspark.sql.functions import col, last, when, rand, avg


def preprocess_train(train, model=None, spark = None):
    if model == 'xgb':
        # train = train.dropna(axis=1, how='all', inplace=False)
        cols = [x for x in train.columns if x not in ['datetime']]
        print('Test Columns : ' + str(len(train.columns)))
        print('Test Rows : ' + str(train.count()))
        train = clip(train, cols)

        # train = train.resample('H').mean()

        train = get_mean_of_cyl_values(train)



        train = train.fillna(0)

        # train.show(n=5)


        return train

    elif model == 'lstm':

        # train = train.dropna(axis=1, how='all', inplace=False)

        cols = [x for x in train.columns if x not in ['datetime']]
        print('Test Columns : ' + str(len(train.columns)))
        print('Test Rows : ' + str(train.count()))
        train = clip(train, cols)

        train = get_mean_of_cyl_values(train)

        # train_max = train.resample('H').max().add_suffix('_max')
        # train_min = train.resample('H').min().add_suffix('_min')
        # train_std = train.resample('H').std().add_suffix('_std')
        # train = train.resample('H').mean()
        #
        # train = pd.concat([train, train_max], axis=1, sort=False)
        # train = pd.concat([train, train_min], axis=1, sort=False)
        # train = pd.concat([train, train_std], axis=1, sort=False)
        # del train_max, train_min,
        # gc.collect()

        # train = train.rolling(window=150).mean()


        cols = [x for x in train.columns if x not in ['datetime']]
        # function to calculate number of seconds from number of days
        days = lambda i: i * 86400
        #
        # train = train.withColumn('datetime', train.datetime.cast('timestamp'))
        #
        # # create window by casting timestamp to long (number of seconds)
        # w = (Window.orderBy('datetime').rowsBetween(-50, 0))
        # for column in cols:
        #     train = train.withColumn(column, avg(train[column]).over(w))

        print('Test Columns : ' + str(len(train.columns)))
        print('Test Rows : ' + str(train.count()))
        print(train.schema)
        train = train.fillna(0)
        #
        # window = Window.orderBy('datetime') \
        #     .rowsBetween(-sys.maxsize, 0)
        #
        # def ffill(column):
        #     return last(column, ignorenulls=True).over(window)
        #
        # def bfill(column):
        #     return last(column, ignorenulls=True).over(window)
        #
        # for column in cols:
        #     train = train.withColumn(column, ffill(col(column)))
        #
        # for column in cols:
        #     train = train.withColumn(column, bfill(col(column)))

        # train = train.fillna(0)
        #
        # vds_5 = train

        # del train
        # gc.collect()
        #
        # vds_5 = vds_5.replace(to_replace=0, value=1)
        #
        # vds_5 = vds_5.pct_change(periods=1, fill_method='ffill')
        # #
        # vds_5 = vds_5.fillna(method='ffill')
        # vds_5 = vds_5.fillna(method='bfill')
        cols = [x for x in train.columns if x not in ['datetime']]
        # vds_55 = normalize(vds_5)
        # vds_55 = scale(vds_55)

        assembler = VectorAssembler().setInputCols \
            (cols).setOutputCol("features")
        print('assembler')
        transformed = assembler.transform(train)


        # Normalize each Vector using $L^1$ norm.
        normalizer = Normalizer(inputCol="features", outputCol="normFeatures", p=1.0)
        l1NormData = normalizer.transform(transformed)

        scaler = StandardScaler(inputCol="normFeatures", outputCol="scaledFeatures",
                                withStd=True, withMean=False)

        # Compute summary statistics by fitting the StandardScaler
        scalerModel = scaler.fit(l1NormData)
        # Normalize each feature to have unit standard deviation.
        scaledData = scalerModel.transform(l1NormData)
        # train = scaledData.drop(*cols)
        del train , transformed,l1NormData

        n_components_ = 50
        # pca = FastICA(n_components=n_components_)
        #
        # dump(pca, 'pca.joblib')
        #
        # pca2_results = pca.fit_transform(scaledData)
        # # n_comp=pca.n_components_
        # n_comp = n_components_
        # print('Number of componeds : ' + str(n_comp))
        # print(pca2_results)
        # print (len(pca2_results[:, 1]))

        # for i in range(0, n_comp):
        #     vds_5['pca_' + str(i)] = 0
        #     # print(len(vds_5['pca_' + str(i)]))
        #     # print(len(pca2_results[:, i]))
        #     vds_5['pca_' + str(i)] = pca2_results[:, i]

        # pca_columns = [x for x in vds_5.columns if x.startswith('pca')]
        # vds_5 = vds_5[pca_columns]

        pca = PCA(k=n_components_, inputCol="scaledFeatures", outputCol="pcaFeatures")
        model = pca.fit(scaledData)

        vds_5 = model.transform(scaledData).select(['pcaFeatures','datetime'])
        print(vds_5)

        def extract(row):
            return (row.datetime,) + tuple(row.pcaFeatures.toArray().tolist())

        vds_5=vds_5.rdd.map(extract).toDF(["datetime"])
        print(vds_5)

        vds_5 = vds_5.drop(*['pcaFeatures', 'datetime'])

        return vds_5

    elif model == 'svm':

        cols = [x for x in train.columns if x not in ['datetime']]
        print('Test Columns : ' + str(len(train.columns)))
        print('Test Rows : ' + str(train.count()))
        train = clip(train, cols)

        print('Test Columns : ' + str(len(train.columns)))
        print('Test Rows : ' + str(train.count()))

        # train_max = train.resample('H').max().add_suffix('_max')
        # train_min = train.resample('H').min().add_suffix('_min')
        # train_std = train.resample('H').std().add_suffix('_std')
        # train = train.resample('H').mean()
        #
        # train = pd.concat([train, train_max], axis=1, sort=False)
        # train = pd.concat([train, train_min], axis=1, sort=False)
        # train = pd.concat([train, train_std], axis=1, sort=False)
        # del train_max, train_min,
        # gc.collect()

        train = get_mean_of_cyl_values(train)

        vds_5 = train
        print('Test Columns : ' + str(len(train.columns)))
        print('Test Rows : ' + str(train.count()))

        # vds_5 = vds_5.replace(to_replace=0, value=1)

        # vds_5 = vds_5.pct_change(periods=1, fill_method='ffill')

        # window = Window.orderBy('datetime') \
        #     .rowsBetween(-sys.maxsize, 0)
        #
        # def ffill(column):
        #     return last(column, ignorenulls=True).over(window)
        #
        # def bfill(column):
        #     return last(column, ignorenulls=True).over(window)
        #
        # for column in cols:
        #     vds_5 = vds_5.withColumn(column, ffill(col(column)))
        #
        # for column in cols:
        #     vds_5 = vds_5.withColumn(column, bfill(col(column)))

        vds_5 = vds_5.fillna(0)
        # vds_5 = vds_5.fillna(method='ffill')
        # vds_5 = vds_5.fillna(method='bfill')

        return vds_5


def preprocess_test(test, model=None):
    # test = test.dropna(axis=1, how='all', inplace=False)
    # for c in test.columns:
    #     if test.filter(col(c).isNotNull()).count() == 0:
    #         test = test.drop(c)

    print('Length of test : ' + str(len(test.columns)))

    if model == 'xgb':
        cols = [x for x in test.columns if x not in ['datetime']]
        print('Test Columns : ' + str(len(test.columns)))
        print('Test Rows : ' + str(test.count()))


        test = clip(test, cols)

        # test = test.resample('H').mean()


        # test = test.rolling(window=50).mean()

        test = get_mean_of_cyl_values(test)
        test = test.fillna(0)

        return test

    elif model == 'lstm':

        cols = [x for x in test.columns if x not in ['datetime']]
        print('Test Columns : ' + str(len(test.columns)))
        print('Test Rows : ' + str(test.count()))
        test = clip(test, cols)

        test = get_mean_of_cyl_values(test)


        print('Test Columns : ' + str(len(test.columns)))
        print('Test Rows : ' + str(test.count()))
        print(test.schema)
        test = test.fillna(0)

        cols = [x for x in test.columns if x not in ['datetime']]


        assembler = VectorAssembler().setInputCols \
            (cols).setOutputCol("features")
        print('assembler')
        transformed = assembler.transform(test)


        # Normalize each Vector using $L^1$ norm.
        normalizer = Normalizer(inputCol="features", outputCol="normFeatures", p=1.0)
        l1NormData = normalizer.transform(transformed)

        scaler = StandardScaler(inputCol="normFeatures", outputCol="scaledFeatures",
                                withStd=True, withMean=False)

        # Compute summary statistics by fitting the StandardScaler
        scalerModel = scaler.fit(l1NormData)
        # Normalize each feature to have unit standard deviation.
        scaledData = scalerModel.transform(l1NormData)
        # train = scaledData.drop(*cols)
        del test , transformed,l1NormData

        n_components_ = 50

        pca = PCA(k=n_components_, inputCol="scaledFeatures", outputCol="pcaFeatures")
        model = pca.fit(scaledData)

        vds_5 = model.transform(scaledData).select(['pcaFeatures','datetime'])
        print(vds_5)

        def extract(row):
            return (row.datetime,) + tuple(row.pcaFeatures.toArray().tolist())

        vds_5=vds_5.rdd.map(extract).toDF(["datetime"])
        print(vds_5)

        vds_5 = vds_5.drop(*['pcaFeatures', 'datetime'])

        return vds_5

    elif model == 'svm':
        # test = test.toPandas()
        # test = clip_data(test)
        cols = [x for x in test.columns if x not in ['datetime']]
        print('Test Columns : ' + str(len(test.columns)))
        print('Test Rows : ' + str(test.count()))


        test = clip(test, cols)

        print('Test Columns : ' + str(len(test.columns)))
        print('Test Rows : ' + str(test.count()))


        # test = test.toPandas()
        # test_max = test.resample('H').max().add_suffix('_max')
        # test_min = test.resample('H').min().add_suffix('_min')
        # test_std = test.resample('H').std().add_suffix('_std')
        # test = test.resample('H').mean()
        #
        # test = pd.concat([test, test_max], axis=1, sort=False)
        # test = pd.concat([test, test_min], axis=1, sort=False)
        # test = pd.concat([test, test_std], axis=1, sort=False)
        # del test_max, test_min,
        # gc.collect()


        # test = test.toHandy()

        test = get_mean_of_cyl_values(test)

        # vds_5 = test
        print('Test Columns : ' + str(len(test.columns)))
        print('Test Rows : ' + str(test.count()))

        # vds_5 = vds_5.replace(to_replace=0, value=1)

        # vds_5 = vds_5.pct_change(periods=1, fill_method='ffill')

        # window = Window.orderBy('datetime') \
        #     .rowsBetween(-sys.maxsize, 0)
        #
        # def ffill(column):
        #     return last(column, ignorenulls=True).over(window)
        #
        # def bfill(column):
        #     return last(column, ignorenulls=True).over(window)
        #
        # for column in cols:
        #     vds_5 = vds_5.withColumn(column,ffill(col(column)))
        #
        # for column in cols:
        #     vds_5 = vds_5.withColumn(column,bfill(col(column)))

        test = test.fillna(0)
        # vds_5 = vds_5.fillna(method='ffill')
        # vds_5 = vds_5.fillna(method='bfill')

        return test

    elif model == 'perm':

        # test = test.resample('H').mean()
        # test = test.rolling(window=20).mean()

        cols = [x for x in test.columns if x not in ['datetime']]
        print('Test Columns : ' + str(len(test.columns)))
        print('Test Rows : ' + str(test.count()))

        test = test.fillna(0)
        test = clip(test, cols)

        # window = Window.orderBy('datetime') \
        #     .rowsBetween(-sys.maxsize, 0)
        #
        # def ffill(column):
        #     return last(column, ignorenulls=True).over(window)
        #
        # def bfill(column):
        #     return last(column, ignorenulls=True).over(window)
        #
        # for column in cols:
        #     test = test.withColumn(column,ffill(col(column)))
        #
        # for column in cols:
        #     test = test.withColumn(column,bfill(col(column)))

        test = test.fillna(0)

        return test
