# // tag::stars-plot[]
ax = (stars_df.toPandas().plot(kind='bar', x='rawStars', y='count', legend=None))

ax.xaxis.set_label_text("")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# end::stars-plot[]

plt.savefig("/tmp/star-count.svg")
plt.close()

# tag::raw-stars-exploration[]
data.describe().select("summary", "rawStars").show()
# end::raw-stars-exploration[]

# tag::raw-stars-exploration-agg[]
data.groupBy("rawStars").agg(F.count("rawStars").alias("count")).sort("rawStars").show()
# end::raw-stars-exploration-agg[]

#  tag::price-range[]
data.groupBy("priceRange").agg(F.count("priceRange").alias("count")).show()
#  end::price-range[]
