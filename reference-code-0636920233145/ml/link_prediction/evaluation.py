# tag::matplotlib-imports[]
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
# end::matplotlib-imports[]


# // tag::feature-importance-plot[]
ax = (feature_importance
 .sort_values("Importance", ascending=False)
 .plot(kind='bar', x='Feature', y='Importance', legend=None))

ax.xaxis.set_label_text("")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# end::feature-importance-plot[]

plt.savefig("/tmp/feature-importance.svg")
plt.close()
