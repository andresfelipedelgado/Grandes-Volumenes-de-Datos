from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import PipelineModel

#Se carga el modelo que previamente se entreno para luego usarlo en la evaluacion de los streams de data
StreamPipeline = PipelineModel.load("Spipeline")

#Se define el esquema de los datos que se leeran en el stream
spark = SparkSession.builder.master("local").appName("Avila-stream").config("spark.some.config.option","some-value").getOrCreate()
schema = "F1 DOUBLE, F2 DOUBLE, F3 DOUBLE, F4 DOUBLE, F5 DOUBLE, F6 DOUBLE, F7 DOUBLE, F8 DOUBLE, F9 DOUBLE, F10 DOUBLE, AuthorNum DOUBLE"

#Definir la variable que va leer los archivos guardados en la carpeta /test/ 
streamingDF = (
  spark
    .readStream
    .schema(schema)
    .option("header","true")
    .option("maxFilesPerTrigger", 1)
    .csv("test/")
)

#Declarar el evaluador que se usara para evaluar las predicciones del modelo
evaluator = MulticlassClassificationEvaluator(labelCol="AuthorNum",predictionCol="prediction", metricName="accuracy")

#Función que se encarga de transformar cada bache y calcular las predicciones de este.
def train_df(df,epoch_id):
    print("----WORKING ON BATCH----")
    print(".................")
    print("# ROWS:",df.count(),", # ATRIBUTES:",len(df.columns))
    prediction = StreamPipeline.transform(df)
    dt_accuracy = evaluator.evaluate(prediction)
    print("----TEST STREAMING RESULTS----")
    print("----BATCH PREDICTIONS---")
    print("Accuracy of RandomForest is = {}" .format(dt_accuracy))

query = streamingDF.writeStream.foreachBatch(train_df).start()   
query.awaitTermination()