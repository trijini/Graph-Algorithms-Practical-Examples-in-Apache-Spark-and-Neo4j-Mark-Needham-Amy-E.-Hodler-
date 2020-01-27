# tag::imports[]
from py2neo import Graph
import pandas as pd

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from pyspark.sql.types import *
from pyspark.sql import functions as F

# end::imports[]


# tag::py2neo[]
graph = Graph("bolt://localhost:7687", auth=("neo4j", "neo"))
# end::py2neo[]


# tag::evaluate-function[]
def evaluate(predictions):
    evaluator = BinaryClassificationEvaluator()
    accuracy = evaluator.evaluate(predictions)
    tp = predictions[(predictions.label == 1) & (predictions.prediction == 1)].count()
    tn = predictions[(predictions.label == 0) & (predictions.prediction == 0)].count()
    fp = predictions[(predictions.label == 0) & (predictions.prediction == 1)].count()
    fn = predictions[(predictions.label == 1) & (predictions.prediction == 0)].count()
    recall = float(tp) / (tp + fn)
    precision = float(tp) / (tp + fp)
    return pd.DataFrame({
        "Measure": ["Accuracy", "Precision", "Recall"],
        "Score": [accuracy, precision, recall]
    })


# end::evaluate-function[]


# tag::prep-function[]
def create_pipeline(fields):
    assembler = VectorAssembler(inputCols=fields, outputCol="features")
    rf = RandomForestClassifier(labelCol="label", featuresCol="features",
                                numTrees=30, maxDepth=10)
    return Pipeline(stages=[assembler, rf])
# end::prep-function[]


# tag::feature-matrix-basic[]
basic_df = graph.run("""
MATCH (p1:Paper)-[link:CITE|:NO_CITATION]->(p2)
WHERE exists((p1)-[:AUTHOR]->()) AND exists((p2)-[:AUTHOR]->())
RETURN p1.id AS p1,
       p2.id AS p2,
       size((p1)-[:AUTHOR]->()<-[:AUTHOR]-(p2)) AS commonAuthors,
       CASE WHEN type(link) = "CITE" THEN 1 ELSE 0 END AS label
""").to_data_frame()

data = spark.createDataFrame(basic_df)
# end::feature-matrix-basic[]

# tag::feature-matrix-columns[]
fields = ['commonAuthors']
# end::feature-matrix-columns[]


# tag::feature-matrix-citations[]
basic_df = graph.run("""
MATCH (p1:Paper)-[link:CITE|:NO_CITATION]->(p2)
WHERE exists((p1)-[:AUTHOR]->()) AND exists((p2)-[:AUTHOR]->())
RETURN p1.id AS p1,
       p2.id AS p2,
       size((p1)-[:AUTHOR]->()<-[:AUTHOR]-(p2)) AS commonAuthors,
       size((p1)<-[:CITE]-()) AS p1Cited,
       size((p2)<-[:CITE]-()) AS p2Cited,
       CASE WHEN type(link) = "CITE" THEN 1 ELSE 0 END AS label
""").to_data_frame()

data = spark.createDataFrame(basic_df)
# end::feature-matrix-citations[]

# tag::feature-matrix-citations-columns[]
fields = ['commonAuthors', 'p1Cited', 'p2Cited']
# end::feature-matrix-citations-columns[]

# tag::feature-matrix-pr[]
basic_df = graph.run("""
MATCH (p1:Paper)-[link:CITE|:NO_CITATION]->(p2)
WHERE exists((p1)-[:AUTHOR]->()) AND exists((p2)-[:AUTHOR]->())
RETURN p1.id AS p1,
       p2.id AS p2,
       size((p1)-[:AUTHOR]->()<-[:AUTHOR]-(p2)) AS commonAuthors,
       size((p1)<-[:CITE]-()) AS p1Cited,
       size((p2)<-[:CITE]-()) AS p2Cited,
       p1.pagerank as p1Pagerank,
       p2.pagerank as p2Pagerank,
       CASE WHEN type(link) = "CITE" THEN 1 ELSE 0 END AS label
""").to_data_frame()

data = spark.createDataFrame(basic_df)
# end::feature-matrix-pr[]

# tag::feature-matrix-pr-columns[]
fields = ['commonAuthors', 'p1Cited', 'p2Cited', 'p1Pagerank', 'p2Pagerank']
# end::feature-matrix-pr-columns[]

# tag::feature-matrix-community[]
df = graph.run("""
MATCH (p1:Paper)-[link:CITE|:NO_CITATION]->(p2)
WHERE exists((p1)-[:AUTHOR]->()) AND exists((p2)-[:AUTHOR]->())
RETURN p1.id AS p1,
       p2.id AS p2,
       size((p1)-[:AUTHOR]->()<-[:AUTHOR]-(p2)) AS commonAuthors,
       size((p1)<-[:CITE]-()) AS p1Cited,
       size((p2)<-[:CITE]-()) AS p2Cited,
       p1.pagerank as p1Pagerank,
       p2.pagerank as p2Pagerank,
       p1.triangles as p1Triangles,
       p2.triangles as p2Triangles,
       p1.coefficient AS p1Coefficient,
       p2.coefficient AS p2Coefficient,
       CASE WHEN p1.louvain = p2.louvain THEN 1 ELSE 0 END AS sameLouvain,
       CASE WHEN type(link) = "CITE" THEN 1 ELSE 0 END AS label
""").to_data_frame()
data = spark.createDataFrame(df)
# end::feature-matrix-community[]


# tag::feature-matrix-community-columns[]
fields = ["commonAuthors", "p1Cited", "p2Cited",
          "p1Pagerank", "p2Pagerank", "samePartition", "p1Triangles", "p2Triangles",
          "p1Coefficient", "p2Coefficient"]
# end::feature-matrix-community-columns[]


# tag::feature-matrix-community-no-coefficient-columns[]
fields = ["commonAuthors", "p1Cited", "p2Cited",
          "p1Pagerank", "p2Pagerank", "samePartition", "p1Triangles", "p2Triangles"]
# end::feature-matrix-community-no-coefficient-columns[]

# tag::feature-matrix-community-no-triangles-columns[]
fields = ["commonAuthors", "p1Cited", "p2Cited",
          "p1Pagerank", "p2Pagerank", "samePartition", "p1Coefficient", "p2Coefficient"]
# end::feature-matrix-community-no-triangles-columns[]

# tag::feature-matrix-community-louvain[]
df = graph.run("""
MATCH (p1:Paper)-[link:CITE|:NO_CITATION]->(p2)
WHERE exists((p1)-[:AUTHOR]->()) AND exists((p2)-[:AUTHOR]->())
RETURN p1.id AS p1,
       p2.id AS p2,
       size((p1)-[:AUTHOR]->()<-[:AUTHOR]-(p2)) AS commonAuthors,
       size((p1)<-[:CITE]-()) AS p1Cited,
       size((p2)<-[:CITE]-()) AS p2Cited,
       p1.pagerank as p1Pagerank,
       p2.pagerank as p2Pagerank,
       p1.triangles as p1Triangles,
       p2.triangles as p2Triangles,
       p1.coefficient AS p1Coefficient,
       p2.coefficient AS p2Coefficient,
       CASE WHEN p1.partition = p2.partition THEN 1 ELSE 0 END AS samePartition,
       CASE WHEN p1.louvain = p2.louvain THEN 1 ELSE 0 END AS sameLouvain,
       CASE WHEN type(link) = "CITE" THEN 1 ELSE 0 END AS label
""").to_data_frame()
data = spark.createDataFrame(df)
# end::feature-matrix-community-louvain[]

# tag::feature-matrix-community-louvain-columns[]
fields = ["commonAuthors", "p1Cited", "p2Cited",
          "p1Pagerank", "p2Pagerank", "samePartition", "sameLouvain",
          "p1Triangles", "p2Triangles", "p1Coefficient", "p2Coefficient"]
# end::feature-matrix-community-louvain-columns[]

# tag::train-test-split[]
(training_data, test_data) = data.randomSplit([0.7, 0.3], 7)
# end::train-test-split[]

# tag::create-pipeline[]
pipeline = create_pipeline(fields)
# end::create-pipeline[]

# tag::train[]
model = pipeline.fit(training_data)
# end::train[]

# tag::predict[]
predictions = model.transform(test_data)
# end::predict[]

# tag::evaluate[]
evaluated_df = evaluate(predictions)
print(evaluated_df)
# end::evaluate[]

# tag::matplotlib-imports[]
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
# end::matplotlib-imports[]

# tag::feature-importance[]
rf_model = model.stages[-1]
feature_importance = pd.DataFrame({"Feature": fields, "Importance": rf_model.featureImportances})
print(feature_importance)
# end::feature-importance[]

# tag::feature-importance-plot-fn[]
def plot_feature_importance(fields, feature_importances):
    df = pd.DataFrame({"Feature": fields, "Importance": feature_importances})
    df = df.sort_values("Importance", ascending=False)
    ax = df.plot(kind='bar', x='Feature', y='Importance', legend=None)
    ax.xaxis.set_label_text("")
    plt.tight_layout()
    plt.show()
# end::feature-importance-plot-fn[]

# tag::feature-importance-plot-fn-file[]
def plot_feature_importance(fields, feature_importances, file):
    df = pd.DataFrame({"Feature": fields, "Importance": feature_importances})
    df = df.sort_values("Importance", ascending=False)
    ax = df.plot(kind='bar', x='Feature', y='Importance', legend=None)
    ax.xaxis.set_label_text("")
    plt.tight_layout()
    plt.savefig(file)
    plt.close()
# end::feature-importance-plot-fn-file[]


# tag::feature-importance-plot[]
rf_model = model.stages[-1]
plot_feature_importance(fields, rf_model.featureImportances)
# end::feature-importance-plot[]

# tag::feature-importance-plot-file[]
rf_model = model.stages[-1]
plot_feature_importance(fields, rf_model.featureImportances, "/tmp/feature-importance-basic.svg")
plot_feature_importance(fields, rf_model.featureImportances, "/tmp/feature-importance-all.svg")
plot_feature_importance(fields, rf_model.featureImportances, "/tmp/feature-importance-louvain.svg")
# end::feature-importance-plot-file[]
