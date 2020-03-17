#!/home/coinlab/anaconda2/envs/spark_env/bin spark_env
# -*- coding: utf-8 -*-
import pickle

from pyspark.ml.classification import GBTClassifier, LinearSVC, RandomForestClassifier, LogisticRegression, \
    DecisionTreeClassifier, NaiveBayes, MultilayerPerceptronClassifier, RandomForestClassificationModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, StringIndexer, VectorIndexer, IndexToString
from pyspark.sql.functions import col, when, rand,lit


class DetectorXGB:
    def __init__(self, data, sc):
        self.data = data
        self.sc = sc

    def fit(self, train):
        from pyspark.ml.feature import MinMaxScaler as minmax
        cols = [x for x in train.columns if x not in ['datetime','label']]

        train = train.fillna(0)
        train = train.withColumn('label', when(rand() > 0.5, 1).otherwise(0))

        print(train.show(n=5))

        assembler = VectorAssembler().setInputCols \
            (cols).setOutputCol("features")

        print('assembler')
        train = assembler.transform(train)
        train = train.fillna(0)
        train = train.drop(*cols)

        rf = RandomForestClassifier(labelCol="label", featuresCol="features", predictionCol='predictions', numTrees=10)

        print('assembler')
        # print(train.show(n=5))
        # train = assembler.transform(train)



        # Chain indexers and forest in a Pipeline
        train.show(n=5)

        # pipeline = Pipeline(stages=[rf])

        print('Train model.  This also runs the indexers.')
        model = rf.fit(train)

        # Save and load model
        model.write().overwrite().save('myRandomForestClassificationModel')
        sameModel = RandomForestClassificationModel.load('myRandomForestClassificationModel')

        print("make predictions")
        # Make predictions.
        predictions = model.transform(train)

        # Select example rows to display.
        predictions.select("predictions", "label", "features").show(5)

        # Select (prediction, true label) and compute test error
        evaluator = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="predictions", metricName="accuracy")
        accuracy = evaluator.evaluate(predictions)
        print("Test Error = %g" % (1.0 - accuracy))


    def predict(self, test):

        cols = [x for x in test.columns if x not in ['datetime', 'label']]
        test = test.fillna(0)
        print(test.printSchema())
        print('Test Columns : ' + str(len(test.columns)))
        print('Test Rows : ' + str(test.count()))

        assembler = VectorAssembler().setInputCols \
            (cols).setOutputCol("features")

        print('assembler')
        test = assembler.transform(test)
        test = test.fillna(0)
        test = test.drop(*cols)


        rf = RandomForestClassificationModel.load('myRandomForestClassificationModel')
        preds = rf.transform(test)
        print(preds.printSchema())
        return preds

