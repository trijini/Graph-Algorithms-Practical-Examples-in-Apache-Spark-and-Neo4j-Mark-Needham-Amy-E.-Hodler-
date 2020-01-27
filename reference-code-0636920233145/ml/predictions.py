# tag::imports[]
from py2neo import Graph
import pandas as pd
from numpy.random import randint

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

# tag::eda[]

# end::eda[]

# tag::evaluate-function[]
def evaluate(predictions):
    evaluator = BinaryClassificationEvaluator()
    accuracy = evaluator.evaluate(predictions)
    tp = predictions[(predictions.label == 1) & (predictions.prediction == 1)].count()
    tn = predictions[(predictions.label == 0) & (predictions.prediction == 0)].count()
    fp = predictions[(predictions.label == 0) & (predictions.prediction == 1)].count()
    fn = predictions[(predictions.label == 1) & (predictions.prediction == 0)].count()
    recall = float(tp)/(tp + fn)
    precision = float(tp) / (tp + fp)
    return pd.DataFrame({
        "Measure": ["Accuracy", "Precision", "Recall"],
        "Score":   [accuracy, precision, recall]
    })
# end::evaluate-function[]

# tag::prep-function[]
def create_pipeline(columns):
    assembler = (VectorAssembler(inputCols=columns, outputCol="features")
        .setHandleInvalid("keep"))
    labelIndexer = StringIndexer(inputCol="stars", outputCol="label")
    rf = RandomForestClassifier(labelCol="label", featuresCol="features",
                                numTrees=30, maxDepth=10)
    return Pipeline(stages=[labelIndexer, assembler, rf])
# end::prep-function[]

# tag::prep-stars-function[]
def create_multi_class_pipeline(columns):
    assembler = VectorAssembler(inputCols=columns, outputCol="features")
    rf = RandomForestClassifier(labelCol="rawStars", featuresCol="features",
                                numTrees=30, maxDepth=10)
    return Pipeline(stages=[assembler, rf])
# end::prep-stars-function[]

# tag::sampling-function[]
def down_sample(data, ratio_adjust=1.2):
    counts = (data.select('stars').groupBy('stars').count()
                  .orderBy(F.column("count"), ascending=False)
                  .collect())
    higher_bound = counts[0][1]
    lower_bound = counts[1][1]
    rand_gen = lambda x: randint(0, higher_bound) if x == "true" else -1
    udf_rand_gen = F.udf(rand_gen, IntegerType())
    threshold_to_filter = int(ratio_adjust * float(lower_bound) / higher_bound * higher_bound)
    data = data.withColumn("randIndex", udf_rand_gen("stars"))
    sampled_training_data = data.filter(data['randIndex'] < threshold_to_filter)
    return sampled_training_data.drop('randIndex')
# end::sampling-function[]

# tag::feature-matrix[]
query = """
MATCH (cat:Category)<-[:IN_CATEGORY]-(business:Business)-[:IN_CITY]->(city:City)
WHERE cat.name = $category AND city.name = $city
MATCH (business:Business)<-[:REVIEWS]-(review:Review)<-[:WROTE]-(user:User)
RETURN user.id AS userId,
       business.id AS businessId,
       review.id AS reviewId,
       size((user)-[:FRIENDS]->()) AS userFriends,
       size((user)-[:WROTE]->()) AS numberOfReviews,
       size((user)-[:WROTE_TIP]->(business)) AS numberOfUserTips,
       size((business)-[:HAS_PHOTO]->()) AS numberOfPhotos,
       size((business)<-[:WROTE_TIP]-()) AS numberOfTips,
       business.priceRange AS priceRange,
       business.delivery AS delivery,
       business.goodForGroups AS goodForGroups,
       business.takeOut AS takeOut,
       review.stars AS rawStars,
       CASE WHEN review.stars > 3 THEN "true" ELSE "false" END as stars
"""

params = {"city": "Las Vegas", "category": "Restaurants"}
basic_df = graph.run(query, params).to_data_frame()
basic_data = spark.createDataFrame(basic_df)
# end::feature-matrix[]

# tag::raw-stars-good-bad[]
data = basic_data.withColumn("goodReview", F.when(basic_data["rawStars"] > 3, "true").otherwise("false"))
# end::raw-stars-good-bad[]

# tag::feature-matrix-users-previous-ratings[]
query = """
MATCH (cat:Category)<-[:IN_CATEGORY]-(business:Business)-[:IN_CITY]->(city:City)
WHERE cat.name = $category AND city.name = $city
MATCH (business:Business)<-[:REVIEWS]-(review:Review)<-[:WROTE]-(user:User)
RETURN user.id AS userId,
	   business.id AS businessId,
       review.id AS reviewId,
       apoc.coll.avg([
         (:Category {name: $category})<-[:IN_CATEGORY]-()<-[:REVIEWS]-(otherRev)<-[:WROTE]-(user)
       	 WHERE otherRev.date < review.date | otherRev.stars]) AS aveOtherStars
"""

user_previous_ratings_df = graph.run(query, {"city": "Las Vegas", "category": "Restaurants"}).to_data_frame()
user_previous_ratings_data = spark.createDataFrame(user_previous_ratings_df)
# end::feature-matrix-users-previous-ratings[]

# tag::feature-matrix-users-previous-ratings-filtered[]
query = """
MATCH (cat:Category)<-[:IN_CATEGORY]-(business:Business)-[:IN_CITY]->(city:City)
WHERE cat.name = $category AND city.name = $city
MATCH (business:Business)<-[:REVIEWS]-(review:Review)<-[:WROTE]-(user:User)
WITH user, business, review, [(otherRev)<-[:WROTE]-(user) WHERE otherRev.date < review.date | otherRev.stars] AS stars
WHERE size(stars) > 0
RETURN user.id AS userId,
	   business.id AS businessId,
       review.id AS reviewId,
       apoc.coll.avg(stars) AS aveOtherStars
"""

params = {"city": "Las Vegas", "category": "Restaurants"}
user_previous_ratings_df = graph.run(query, params).to_data_frame()
user_previous_ratings_data = spark.createDataFrame(user_previous_ratings_df)
# end::feature-matrix-users-previous-ratings-filtered[]

# tag::feature-matrix-user-previous-ratings-join[]
data = data.join(user_previous_ratings_data, ["businessId", "userId", "reviewId"])
# end::feature-matrix-user-previous-ratings-join[]

# tag::feature-matrix-columns[]
columns = ['userFriends', 'numberOfReviews', 'numberOfPhotos', 'priceRange',
           'numberOfUserTips', 'aveOtherStars']
# end::feature-matrix-columns[]

# tag::show-feature-matrix[]
data.select(columns + ["stars"]).show(truncate=False)
# end::show-feature-matrix[]

# tag::feature-matrix-friends-ratings[]
# Takes a long time to run
import time
t0 = time.clock()
query = """
MATCH (cat:Category)<-[:IN_CATEGORY]-(business:Business)-[:IN_CITY]->(city:City)
WHERE cat.name = $category AND city.name = $city
MATCH (business:Business)<-[:REVIEWS]-(review:Review)<-[:WROTE]-(user:User)
RETURN review.id AS reviewId,
       business.id AS businessId,
       user.id AS userId,
       apoc.coll.avg([
         (business)<-[:REVIEWS]-(otherRev)<-[:WROTE]-(other:User)-[:FRIENDS]->(user)
       	 WHERE otherRev.date < review.date | otherRev.stars]) AS aveFriendsStars
"""
friends_df = graph.run(query, {"city": "Las Vegas", "category": "Restaurants"}).to_data_frame()
friends_data = spark.createDataFrame(friends_df)
run_time = time.clock() - t0
print(run_time)
# end::feature-matrix-friends-ratings[]

# tag::feature-matrix-business-ranking-influential[]
query = """
MATCH (cat:Category)<-[:IN_CATEGORY]-(business:Business)-[:IN_CITY]->(city:City)
WHERE cat.name = $category AND city.name = $city
MATCH (business:Business)<-[:REVIEWS]-(review:Review)<-[:WROTE]-(user:User)
WHERE user.restaurantsPageRank > $cutOff
RETURN business.id AS businessId, avg(review.stars) as aveInfluentialRating
"""

params = {"city": "Las Vegas", "category": "Restaurants", "cutOff": 1.50}
business_influence_df = graph.run(query, params).to_data_frame()
business_influence_data = spark.createDataFrame(business_influence_df)
# end::feature-matrix-business-ranking-influential[]

# tag::feature-matrix-business-ranking-influential-join[]
data = data.join(business_influence_data, "businessId")
# end::feature-matrix-business-ranking-influential-join[]

# tag::feature-matrix-graph-columns[]
columns = ['userFriends', 'numberOfReviews', 'numberOfPhotos', 'priceRange',
           'numberOfUserTips', 'aveOtherStars', 'aveInfluentialRating']
# end::feature-matrix-graph-columns[]

# The rating of people in the same cluster...

# tag::train-test-split[]
(training_data, test_data) = data.randomSplit([0.7, 0.3], 7)
# end::train-test-split[]

# tag::sampling[]
sampled_training_data = down_sample(training_data)
# end::sampling[]

# tag::prep[]
pipeline = create_pipeline(columns)
pipeline = create_multi_class_pipeline(columns)
# end::prep[]

# tag::create-pipeline[]
pipeline = create_pipeline(columns)
# end::create-pipeline[]

# tag::create-multiclass-pipeline[]
pipeline = create_multi_class_pipeline(columns)
# end::create-multiclass-pipeline[]

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

# tag::evaluate-multiclass[]
predictions = predictions.withColumn("rawStars", predictions["rawStars"].cast(DoubleType()))
metrics = MulticlassMetrics(predictions.select("prediction", "rawStars").rdd)
# end::evaluate-multiclass[]

# tag::evaluate-multiclass-summary[]
summary_stats = pd.DataFrame({
    "Measure": ["Precision", "Recall", "F1 Score",
                "Weighted Precision", "Weighted Recall", "Weighted F(1) Score",
                "Weighted F(0.5) Score", "Weighted false positive rate"],
    "Score":   [metrics.precision(), metrics.recall(), metrics.fMeasure(),
                metrics.weightedPrecision, metrics.weightedRecall, metrics.weightedFMeasure(),
                metrics.weightedFMeasure(beta=0.5), metrics.weightedFalsePositiveRate]
})
print(summary_stats)
# end::evaluate-multiclass-summary[]

# tag::evaluate-multiclass-summary-by-label[]
labels = [1.0, 2.0, 3.0, 4.0, 5.0]
summary_stats_by_label = pd.DataFrame({
    "Label": labels,
    "Precision": [metrics.precision(label) for label in labels],
    "Recall": [metrics.recall(label) for label in labels],
    "F1 Score": [metrics.fMeasure(label, beta=1.0) for label in labels]
})
print(summary_stats_by_label)
# end::evaluate-multiclass-summary-by-label[]

# tag::model-exploration[]
rf_model = model.stages[-1]
print(pd.DataFrame({"Feature": columns, "Importance": rf_model.featureImportances}))
# end::model-exploration[]

# tag::predict-exploration[]
predictions.select("label", "prediction", "features").show(10,truncate=False)
predictions.groupBy("prediction").agg(F.count("prediction")).show()
# end::predict-exploration[]
