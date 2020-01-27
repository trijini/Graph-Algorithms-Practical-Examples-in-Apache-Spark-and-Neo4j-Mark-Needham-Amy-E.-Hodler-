# tag::label[]
data.describe().select("summary", "label").show()
# end::label[]

# tag::label-agg[]
data.groupBy("label").agg(F.count("label").alias("count")).show()
# end::label-agg[]

# tag::common-authors[]
data.stat.corr("commonAuthors", "label")
# end::common-authors[]


# tag::basic-correlations[]
data.select("commonAuthors", "p1Cited", "p2Cited", "label").toPandas().corr()
# end::basic-correlations[]


# tag::visualize-decision-tree[]
from spark_tree_plotting import export_graphviz

dot_string = export_graphviz(rf_model.trees[0],
    featureNames=fields, categoryNames=[], classNames=["True", "False"],
    filled=True, roundedCorners=True, roundLeaves=True)

with open("/tmp/rf.dot", "w") as file:
    file.write(dot_string)
# end::visualize-decision-tree[]
