from scripts.ml.link_prediction_aminer.reqs import *

# tag::down-sample[]
def down_sample(df):
    copy = df.copy()
    zero = Counter(copy.label.values)[0]
    un = Counter(copy.label.values)[1]
    n = zero - un
    copy = copy.drop(copy[copy.label == 0].sample(n=n, random_state=1).index)
    return copy.sample(frac=1)
# end::down-sample[]


# tag::apply-train-features-graphy-fn[]
def apply_graphy_training_features(data):
    query = """
    UNWIND $pairs AS pair
    MATCH (p1) WHERE id(p1) = pair.node1
    MATCH (p2) WHERE id(p2) = pair.node2
    RETURN pair.node1 AS node1,
           pair.node2 AS node2,
           size([(p1)-[:CO_AUTHOR_EARLY]-(a)-[:CO_AUTHOR_EARLY]-(p2) | a]) AS commonAuthors,
           size((p1)-[:CO_AUTHOR_EARLY]-()) * size((p2)-[:CO_AUTHOR_EARLY]-()) AS prefAttachment,
           size(apoc.coll.toSet(
             [(p1)-[:CO_AUTHOR_EARLY]-(a) | id(a)] + [(p2)-[:CO_AUTHOR_EARLY]-(a) | id(a)]
           )) AS totalNeighbors
    """
    pairs = [{"node1": row["node1"], "node2": row["node2"]} for row in data.collect()]
    features = spark.createDataFrame(graph.run(query, {"pairs": pairs}).to_data_frame())
    return data.join(features, ["node1", "node2"])
# end::apply-train-features-graphy-fn[]

# tag::apply-test-features-graphy-fn[]
def apply_graphy_test_features(data):
    query = """
    UNWIND $pairs AS pair
    MATCH (p1) WHERE id(p1) = pair.node1
    MATCH (p2) WHERE id(p2) = pair.node2
    RETURN pair.node1 AS node1,
           pair.node2 AS node2,
           size([(p1)-[:CO_AUTHOR]-(a)-[:CO_AUTHOR]-(p2) | a]) AS commonAuthors,
           size((p1)-[:CO_AUTHOR]-()) * size((p2)-[:CO_AUTHOR]-()) AS prefAttachment,
           size(apoc.coll.toSet(
             [(p1)-[:CO_AUTHOR]-(a) | id(a)] + [(p2)-[:CO_AUTHOR]-(a) | id(a)]
           )) AS totalNeighbors
    """
    pairs = [{"node1": row["node1"], "node2": row["node2"]} for row in data.collect()]
    features = spark.createDataFrame(graph.run(query, {"pairs": pairs}).to_data_frame())
    return data.join(features, ["node1", "node2"])
# end::apply-test-features-graphy-fn[]


# tag::apply-triangles-features-fn[]
def apply_triangles_features(data, triangles_prop, coefficient_prop):
    query = """
    UNWIND $pairs AS pair
    MATCH (p1) WHERE id(p1) = pair.node1
    MATCH (p2) WHERE id(p2) = pair.node2
    RETURN pair.node1 AS node1,
           pair.node2 AS node2,
           apoc.coll.min([p1[$trianglesProp], p2[$trianglesProp]]) AS minTriangles,
           apoc.coll.max([p1[$trianglesProp], p2[$trianglesProp]]) AS maxTriangles,
           apoc.coll.min([p1[$coefficientProp], p2[$coefficientProp]]) AS minCoefficient,
           apoc.coll.max([p1[$coefficientProp], p2[$coefficientProp]]) AS maxCoefficient
    """
    params = {
        "pairs": [{"node1": row["node1"], "node2": row["node2"]} for row in data.collect()],
        "trianglesProp": triangles_prop,
        "coefficientProp": coefficient_prop
    }
    features = spark.createDataFrame(graph.run(query, params).to_data_frame())
    return data.join(features, ["node1", "node2"])
# end::apply-triangles-features-fn[]

# tag::apply-features-community-fn[]
def apply_community_features(data, partition_prop, louvain_prop):
    query = """
    UNWIND $pairs AS pair
    MATCH (p1) WHERE id(p1) = pair.node1
    MATCH (p2) WHERE id(p2) = pair.node2
    RETURN pair.node1 AS node1,
           pair.node2 AS node2,
           CASE WHEN p1[$partitionProp] = p2[$partitionProp] THEN 1 ELSE 0 END AS samePartition,
           CASE WHEN p1[$louvainProp] = p2[$louvainProp] THEN 1 ELSE 0 END AS sameLouvain
    """
    params = {
        "pairs": [{"node1": row["node1"], "node2": row["node2"]} for row in data.collect()],
        "partitionProp": partition_prop,
        "louvainProp": louvain_prop
    }
    features = spark.createDataFrame(graph.run(query, params).to_data_frame())
    return data.join(features, ["node1", "node2"])
# end::apply-features-community-fn[]

# tag::create-pipeline-fn[]
def create_pipeline(fields):
    assembler = VectorAssembler(inputCols=fields, outputCol="features")
    rf = RandomForestClassifier(labelCol="label", featuresCol="features",
                                numTrees=30, maxDepth=10)
    return Pipeline(stages=[assembler, rf])
# end::create-pipeline-fn[]


# tag::train-model-fn[]
def train_model(fields, training_data):
    pipeline = create_pipeline(fields)
    model = pipeline.fit(training_data)
    return model
# end::train-model-fn[]

# tag::evaluate2-fn[]
def evaluate_model(model, test_data):
    # Execute the model against the test set
    predictions = model.transform(test_data)

    # Compute true positive, false positive, false negative counts
    tp = predictions[(predictions.label == 1) & (predictions.prediction == 1)].count()
    fp = predictions[(predictions.label == 0) & (predictions.prediction == 1)].count()
    fn = predictions[(predictions.label == 1) & (predictions.prediction == 0)].count()

    # Compute recall and precision manually
    recall = float(tp) / (tp + fn)
    precision = float(tp) / (tp + fp)

    # Compute accuracy using Spark MLLib's binary classification evaluator
    accuracy = BinaryClassificationEvaluator().evaluate(predictions)

    # Compute false positive rate and true positive rate using sklearn functions
    labels = [row["label"] for row in predictions.select("label").collect()]
    preds = [row["probability"][1] for row in predictions.select("probability").collect()]
    fpr, tpr, threshold = roc_curve(labels, preds)
    roc_auc = auc(fpr, tpr)

    return { "fpr": fpr, "tpr": tpr, "roc_auc": roc_auc, "accuracy": accuracy,
             "recall": recall, "precision": precision }


# end::evaluate2-fn[]

# tag::display-eval-results-fn[]
def display_results(results):
    results = {k: v for k, v in results.items() if k not in ["fpr", "tpr", "roc_auc"]}
    return pd.DataFrame({"Measure": list(results.keys()), "Score": list(results.values())})
# end::display-eval-results-fn[]


# tag::plot-auc-curve-fn[]
def create_roc_plot():
    plt.style.use('classic')
    fig = plt.figure(figsize=(13, 8))
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'c', 'm', 'y', 'k'])))
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random score (AUC = 0.50)')
    return plt, fig


def add_curve(plt, title, fpr, tpr, roc):
    plt.plot(fpr, tpr, label=f"{title} (AUC = {roc:0.2})")
# end::plot-auc-curve-fn[]


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
def plot_feature_importance_file(fields, feature_importances, file):
    plt.style.use('fivethirtyeight')
    df = pd.DataFrame({"Feature": fields, "Importance": feature_importances})
    df = df.sort_values("Importance", ascending=False)
    ax = df.plot(kind='bar', x='Feature', y='Importance', legend=None)
    ax.xaxis.set_label_text("")
    plt.tight_layout()
    plt.savefig(file)
    plt.close()
# end::feature-importance-plot-fn-file[]


# tag::train-set[]
train_existing_links = graph.run("""
MATCH (author:Author)-[:CO_AUTHOR_EARLY]->(other:Author)
RETURN id(author) AS node1, id(other) AS node2, 1 AS label
""").to_data_frame()

train_missing_links = graph.run("""
MATCH (author:Author)
WHERE (author)-[:CO_AUTHOR_EARLY]-()
MATCH (author)-[:CO_AUTHOR_EARLY*2..3]-(other)
WHERE not((author)-[:CO_AUTHOR_EARLY]-(other))
RETURN id(author) AS node1, id(other) AS node2, 0 AS label
""").to_data_frame()

train_missing_links = train_missing_links.drop_duplicates()
training_df = train_missing_links.append(train_existing_links, ignore_index=True)
training_df['label'] = training_df['label'].astype('category')
training_df = down_sample(training_df)
training_data = spark.createDataFrame(training_df)
# end::train-set[]

# tag::check-train-set[]
training_data.groupby("label").count().show()
# end::check-train-set[]

# tag::explore-data[]
training_data.show(n=5)
# end::explore-data[]

# tag::test-set[]
test_existing_links = graph.run("""
MATCH (author:Author)-[:CO_AUTHOR_LATE]->(other:Author)
RETURN id(author) AS node1, id(other) AS node2, 1 AS label
""").to_data_frame()

test_missing_links = graph.run("""
MATCH (author:Author)
WHERE (author)-[:CO_AUTHOR_LATE]-()
MATCH (author)-[:CO_AUTHOR*2..3]-(other)
WHERE not((author)-[:CO_AUTHOR]-(other))
RETURN id(author) AS node1, id(other) AS node2, 0 AS label
""").to_data_frame()

test_missing_links = test_missing_links.drop_duplicates()
test_df = test_missing_links.append(test_existing_links, ignore_index=True)
test_df['label'] = test_df['label'].astype('category')
test_df = down_sample(test_df)
test_data = spark.createDataFrame(test_df)
# end::test-set[]

# tag::check-test-set[]
test_data.groupby("label").count().show()
# end::check-test-set[]

############
# EXPLORATION
############

# tag::explore-common-authors[]
plt.style.use('fivethirtyeight')
fig, axs = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
charts = [(1, "have collaborated"), (0, "haven't collaborated")]

for index, chart in enumerate(charts):
    label, title = chart
    filtered = training_data.filter(training_data["label"] == label)
    common_authors = filtered.toPandas()["commonAuthors"]
    histogram = common_authors.value_counts().sort_index()
    histogram /= float(histogram.sum())
    histogram.plot(kind="bar", x='Common Authors', color="darkblue", ax=axs[index],
                   title=f"Authors who {title} (label={label})")
    axs[index].xaxis.set_label_text("Common Authors")

plt.tight_layout()
plt.show()
# end::explore-common-authors[]

# tag::explore-graphy-features-positive[]
(training_data.filter(training_data["label"]==1)
 .describe()
 .select("summary", "commonAuthors", "prefAttachment", "totalNeighbors")
 .show())
# end::explore-graphy-features-positive[]

# tag::explore-graphy-features-negative[]
(training_data.filter(training_data["label"]==0)
 .describe()
 .select("summary", "commonAuthors", "prefAttachment", "totalNeighbors")
 .show())
# end::explore-graphy-features-negative[]

# tag::explore-triangles-features-positive[]
(training_data.filter(training_data["label"]==1)
 .describe()
 .select("summary", "minTriangles", "maxTriangles", "minCoefficient", "maxCoefficient")
 .show())
# end::explore-triangles-features-positive[]

# tag::explore-triangles-features-negative[]
(training_data.filter(training_data["label"]==0)
 .describe()
 .select("summary", "minTriangles", "maxTriangles", "minCoefficient", "maxCoefficient")
 .show())
# end::explore-triangles-features-negative[]

# tag::explore-community-partition[]
plt.style.use('fivethirtyeight')
fig, axs = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
charts = [(1, "have collaborated"), (0, "haven't collaborated")]

for index, chart in enumerate(charts):
    label, title = chart
    filtered = training_data.filter(training_data["label"] == label)
    values = (filtered.withColumn('samePartition', F.when(F.col("samePartition") == 0, "False")
                                  .otherwise("True"))
              .groupby("samePartition")
              .agg(F.count("label").alias("count"))
              .select("samePartition", "count")
              .toPandas())
    values.set_index("samePartition", drop=True, inplace=True)
    values.plot(kind="bar", ax=axs[index], legend=None,
                title=f"Authors who {title} (label={label})")
    axs[index].xaxis.set_label_text("Same Partition")

plt.tight_layout()
plt.show()
# end::explore-community-partition[]

# tag::explore-community-louvain[]
plt.style.use('fivethirtyeight')
fig, axs = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
charts = [(1, "have collaborated"), (0, "haven't collaborated")]

for index, chart in enumerate(charts):
    label, title = chart
    filtered = training_data.filter(training_data["label"] == label)
    values = (filtered.withColumn('sameLouvain', F.when(F.col("sameLouvain") == 0, "False")
                                  .otherwise("True"))
              .groupby("sameLouvain")
              .agg(F.count("label").alias("count"))
              .select("sameLouvain", "count")
              .toPandas())
    values.set_index("sameLouvain", drop=True, inplace=True)
    values.plot(kind="bar", ax=axs[index], legend=None,
                title=f"Authors who {title} (label={label})")
    axs[index].xaxis.set_label_text("Same Louvain")

plt.tight_layout()
plt.show()
# end::explore-community-louvain[]

############
# BASIC
############

# tag::apply-features[]
training_data = apply_graphy_training_features(training_data)
test_data = apply_graphy_test_features(test_data)
# end::apply-features[]

# tag::train-model-basic[]
basic_model = train_model(["commonAuthors"], training_data)
# end::train-model-basic[]

# tag::evaluate-model-basic[]
basic_results = evaluate_model(basic_model, test_data)
display_results(basic_results)
# end::evaluate-model-basic[]

# tag::plot-auc-curve-basic[]
plt, fig = create_roc_plot()

add_curve(plt, "Common Authors",
          basic_results["fpr"], basic_results["tpr"], basic_results["roc_auc"])

plt.legend(loc='lower right')
plt.show()
# end::plot-auc-curve-basic[]
fig.savefig('/tmp/roc_basic.svg')

# tag::test-model-manually[]
eval_df = spark.createDataFrame(
    [(0,), (1,), (2,), (10,), (100,)],
    ['commonAuthors'])

(basic_model.transform(eval_df)
 .select("commonAuthors", "probability", "prediction")
 .show(truncate=False))
# end::test-model-manually[]

############
# GRAPHY
############

# tag::train-model-graphy[]
fields = ["commonAuthors", "prefAttachment", "totalNeighbors"]
graphy_model = train_model(fields, training_data)
# end::train-model-graphy[]

# tag::evaluate-model-graphy[]
graphy_results = evaluate_model(graphy_model, test_data)
display_results(graphy_results)
# end::evaluate-model-graphy[]

# tag::plot-auc-curve-graphy[]
plt, fig = create_roc_plot()

add_curve(plt, "Common Authors",
          basic_results["fpr"], basic_results["tpr"], basic_results["roc_auc"])

add_curve(plt, "Graphy",
          graphy_results["fpr"], graphy_results["tpr"], graphy_results["roc_auc"])

plt.legend(loc='lower right')
plt.show()
# end::plot-auc-curve-graphy[]
fig.savefig('/tmp/roc_graph.svg')

# tag::feature-importance-graphy-plot[]
rf_model = graphy_model.stages[-1]
plot_feature_importance(fields, rf_model.featureImportances)
# end::feature-importance-graphy-plot[]

# tag::feature-importance-graphy-plot-file[]
rf_model = graphy_model.stages[-1]
plot_feature_importance_file(fields, rf_model.featureImportances,
                             "/tmp/feature-importance-basic.svg")
# end::feature-importance-graphy-plot-file[]

############
# TRIANGLES
############

# tag::apply-triangle-features[]
training_data = apply_triangles_features(training_data, "trianglesTrain", "coefficientTrain")
test_data = apply_triangles_features(test_data, "trianglesTest", "coefficientTest")
# end::apply-triangle-features[]

# tag::train-model-triangles[]
fields = ["commonAuthors", "prefAttachment", "totalNeighbors",
          "minTriangles", "maxTriangles", "minCoefficient", "maxCoefficient"]
triangle_model = train_model(fields, training_data)
# end::train-model-triangles[]

# tag::evaluate-model-triangles[]
triangle_results = evaluate_model(triangle_model, test_data)
display_results(triangle_results)
# end::evaluate-model-triangles[]

# tag::plot-auc-curve-triangles[]
plt, fig = create_roc_plot()

add_curve(plt, "Common Authors",
          basic_results["fpr"], basic_results["tpr"], basic_results["roc_auc"])

add_curve(plt, "Graphy",
          graphy_results["fpr"], graphy_results["tpr"], graphy_results["roc_auc"])

add_curve(plt, "Triangles",
          triangle_results["fpr"], triangle_results["tpr"], triangle_results["roc_auc"])

plt.legend(loc='lower right')
plt.show()
# end::plot-auc-curve-triangles[]
fig.savefig('/tmp/roc_triangles.svg')

# tag::feature-importance-triangles-plot[]
rf_model = triangle_model.stages[-1]
plot_feature_importance(fields, rf_model.featureImportances)
# end::feature-importance-triangles-plot[]

# tag::feature-importance-triangles-plot-file[]
rf_model = triangle_model.stages[-1]
plot_feature_importance_file(fields, rf_model.featureImportances,
                             "/tmp/feature-importance-triangles.svg")
# end::feature-importance-triangles-plot-file[]

############
# COMMUNITY
############

# tag::apply-community-features[]
training_data = apply_community_features(training_data, "partitionTrain", "louvainTrain")
test_data = apply_community_features(test_data, "partitionTest", "louvainTest")
# end::apply-community-features[]

# tag::train-model-community[]
fields = ["commonAuthors", "prefAttachment", "totalNeighbors",
          "minTriangles", "maxTriangles", "minCoefficient", "maxCoefficient",
          "samePartition", "sameLouvain"]
community_model = train_model(fields, training_data)
# end::train-model-community[]

# tag::evaluate-model-community[]
community_results = evaluate_model(community_model, test_data)
display_results(community_results)
# end::evaluate-model-community[]

# tag::plot-auc-curve-community[]
plt, fig = create_roc_plot()

add_curve(plt, "Common Authors",
          basic_results["fpr"], basic_results["tpr"], basic_results["roc_auc"])

add_curve(plt, "Graphy",
          graphy_results["fpr"], graphy_results["tpr"], graphy_results["roc_auc"])

add_curve(plt, "Triangles",
          triangle_results["fpr"], triangle_results["tpr"], triangle_results["roc_auc"])

add_curve(plt, "Community",
          community_results["fpr"], community_results["tpr"], community_results["roc_auc"])

plt.legend(loc='lower right')
plt.show()
# end::plot-auc-curve-community[]
fig.savefig('/tmp/roc_community.svg')

# tag::feature-importance-community-plot[]
rf_model = community_model.stages[-1]
plot_feature_importance(fields, rf_model.featureImportances)
# end::feature-importance-community-plot[]

# tag::feature-importance-community-plot-file[]
rf_model = community_model.stages[-1]
plot_feature_importance_file(fields, rf_model.featureImportances,
                             "/tmp/feature-importance-community.svg")
# end::feature-importance-community-plot-file[]
