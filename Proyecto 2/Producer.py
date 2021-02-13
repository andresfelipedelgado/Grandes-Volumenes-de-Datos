from pyspark.sql import SparkSession
from pyspark.sql.functions import count, col, explode, array, lit
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler, VectorSizeHint
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
#Se carga el conjunto de datos

spark = SparkSession.builder.master("local").appName("Avila").config("spark.some.config.option","some-value").getOrCreate()
data = spark.read.format("csv").option("header","true").option("inferSchema", "true").load(r"avila.csv")

#####################################################################################################
#LIMPIEZA DE LOS DATOS
#Eliminacion de atributos muy atipicos de F2
data = data.filter(data.F2<350)

indexer = StringIndexer(inputCol="Author", outputCol="AuthorNum")
data = indexer.fit(data).transform(data)
data = data.drop('Author')
data.groupby("AuthorNum").count()

#Se balancea cada categoria (Entre 1000 y 2000 atributos cada una)
A = data.filter(data.AuthorNum == 0.0).sample(fraction=0.24)
F = data.filter(data.AuthorNum == 1.0).sample(fraction=0.53)
E = data.filter(col("AuthorNum") == 2.0).withColumn("dummy", explode(array([lit(x) for x in range(1)]))).drop('dummy')
I = data.filter(col("AuthorNum") == 3.0).withColumn("dummy", explode(array([lit(x) for x in range(1)]))).drop('dummy')
X = data.filter(col("AuthorNum") == 4.0).withColumn("dummy", explode(array([lit(x) for x in range(2)]))).drop('dummy')
H = data.filter(col("AuthorNum") == 5.0).withColumn("dummy", explode(array([lit(x) for x in range(2)]))).drop('dummy')
G = data.filter(col("AuthorNum") == 6.0).withColumn("dummy", explode(array([lit(x) for x in range(2)]))).drop('dummy')
D = data.filter(col("AuthorNum") == 7.0).withColumn("dummy", explode(array([lit(x) for x in range(3)]))).drop('dummy')
Y = data.filter(col("AuthorNum") == 8.0).withColumn("dummy", explode(array([lit(x) for x in range(4)]))).drop('dummy')
C = data.filter(col("AuthorNum") == 9.0).withColumn("dummy", explode(array([lit(x) for x in range(8)]))).drop('dummy')
W = data.filter(col("AuthorNum") == 10.0).withColumn("dummy", explode(array([lit(x) for x in range(17)]))).drop('dummy')
B = data.filter(col("AuthorNum") == 11.0).withColumn("dummy", explode(array([lit(x) for x in range(170)]))).drop('dummy')

#Se juntan todas las categorias balanceadas
data = A.union(B).union(C).union(D).union(E).union(F).union(G).union(H).union(I).union(W).union(Y).union(X)

data.groupby("AuthorNum").count()
print("Numero de Registros Dataset Limpio:",data.count(),", Atributos:",len(data.columns))

#####################################################################################################
#Se crea el conjunto de entrenamiento y test
train, test = data.randomSplit([0.7, 0.3],seed=20)
print("ENTRENAMIENTO:")
print("Numero de Registros Train:",train.count(),", Atributos:",len(train.columns))
print("TEST:")
print("Numero de Registros Test:",test.count(),", Atributos:",len(test.columns))

#creación baches
num_baches = 4
div = 1/num_baches

listDiv = []
for i in range(num_baches):
	listDiv.append(div)
dfAns = test.randomSplit(listDiv,seed=20)
for j in range(len(dfAns)):
	dfAns[j].toPandas().to_csv('test/{}.csv'.format(j+1),index=False)

#Crear al pipeline con dos etapas
cols=data.columns
cols.remove("AuthorNum")
assembler = VectorAssembler(inputCols=cols,outputCol="features")
rf = RandomForestClassifier(labelCol="AuthorNum", featuresCol="features",numTrees=10,subsamplingRate=1,maxDepth=10)
pipeline = Pipeline(stages=[assembler,rf])

#Guardar el pipeline en la carpeta Spipeline
pipelineModel = pipeline.fit(train)
pipelineModel.write().overwrite().save("Spipeline")