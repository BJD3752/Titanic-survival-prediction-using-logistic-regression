# Databricks notebook source
import pyspark

# COMMAND ----------

df = spark.sql('select * from titanic_csv')

# COMMAND ----------

df.show()

# COMMAND ----------

#df.printSchema()
df.columns

# COMMAND ----------

my_cols = df.select(['Survived','Pclass',
                     'Sex','Age',
                     'SibSp','Parch',
                     'Fare','Embarked'])

# COMMAND ----------

#dropping the missing datas
my_final_data = my_cols.na.drop()
my_final_data.columns

# COMMAND ----------

from pyspark.ml.feature import (VectorAssembler,VectorIndexer,
                               OneHotEncoder,StringIndexer)

# COMMAND ----------

#index sex male, female to index
gender_indexer = StringIndexer(inputCol = 'Sex',outputCol = 'SexIndex')
# A B C
# 0 1 2 - indexing
#One Hot Encode
# Key A B C
# example for a it will be [1 , 0, 0]

#OneHoEncodeing- make a vector from indexed column
gender_encoder = OneHotEncoder(inputCol ='SexIndex', outputCol= 'SexVec')

# COMMAND ----------

embark_indexer = StringIndexer(inputCol='Embarked',outputCol='EmbarkedIndex')
embark_encoder = OneHotEncoder(inputCol='EmbarkedIndex',outputCol= 'EmbarkedVec')

# COMMAND ----------

#assembled all col into ine vector col features
assembler = VectorAssembler(inputCols=['Pclass','SexVec','EmbarkedVec','Age','SibSp','Parch','Fare'],
                           outputCol='features')

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression

# COMMAND ----------

log_reg_titanic = LogisticRegression(featuresCol='features',labelCol='Survived')

# COMMAND ----------

from pyspark.ml import Pipeline

# COMMAND ----------

#creating a pipeline stages for better execution code.
pipeline = Pipeline(stages=[gender_indexer,embark_indexer,
                            gender_encoder,embark_encoder,
                            assembler,log_reg_titanic])


# COMMAND ----------

train_data,test_data = my_final_data.randomSplit([0.7,0.3])

# COMMAND ----------

fit_model = pipeline.fit(train_data)

# COMMAND ----------

results = fit_model.transform(test_data)
results.select('survived','prediction').show()

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator

# COMMAND ----------

my_evel = BinaryClassificationEvaluator(rawPredictionCol='prediction',labelCol='Survived')

# COMMAND ----------

#AUC(area under curve):
AUC = my_evel.evaluate(results)
print(AUC)

# COMMAND ----------

#Our result will 75% accurate to original survived list

# COMMAND ----------


