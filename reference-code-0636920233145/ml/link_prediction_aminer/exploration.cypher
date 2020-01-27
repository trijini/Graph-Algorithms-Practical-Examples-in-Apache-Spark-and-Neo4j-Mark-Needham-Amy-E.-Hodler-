
// tag::train-test[]
MATCH (article:Article)
RETURN article.year < 2006 AS training, count(*) AS count
// end::train-test[]

// tag::early[]
MATCH ()-[:CO_AUTHOR_EARLY]->()
RETURN count(*) AS count
// end::early[]


// tag::late[]
MATCH ()-[:CO_AUTHOR_LATE]->()
RETURN count(*) AS count
// end::late[]
