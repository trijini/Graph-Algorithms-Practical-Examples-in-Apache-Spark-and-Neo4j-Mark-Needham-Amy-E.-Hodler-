// tag::schema[]
CREATE CONSTRAINT ON (article:Article)
ASSERT article.index IS UNIQUE;

CREATE CONSTRAINT ON (author:Author)
ASSERT author.name IS UNIQUE;
// end::schema[]

// tag::import[]
CALL apoc.periodic.iterate(
  'UNWIND ["dblp-ref-0.json","dblp-ref-1.json","dblp-ref-2.json","dblp-ref-3.json"] AS file
   CALL apoc.load.json("file:///" + file)
   YIELD value
   WHERE value.venue IN ["Lecture Notes in Computer Science", "Communications of The ACM",
                         "international conference on software engineering",
                         "advances in computing and communications"]
   return value',
  'MERGE (a:Article {index:value.id})
   ON CREATE SET a += apoc.map.clean(value,["id","authors","references"],[0])
   WITH a,value.authors as authors
   UNWIND authors as author
   MERGE (b:Author{name:author})
   MERGE (b)<-[:AUTHOR]-(a)'
, {batchSize: 10000, iterateList: true});
// end::import[]

// tag::co-author[]
MATCH (a1)<-[:AUTHOR]-(paper)-[:AUTHOR]->(a2:Author)
WITH a1, a2, paper
ORDER BY a1, paper.year
WITH a1, a2, collect(paper)[0].year AS year, count(*) AS collaborations
MERGE (a1)-[coauthor:CO_AUTHOR {year: year}]-(a2)
SET coauthor.collaborations = collaborations;
// end::co-author[]

// tag::co-author-early[]
MATCH (a1)<-[:AUTHOR]-(paper)-[:AUTHOR]->(a2:Author)
WITH a1, a2, paper
ORDER BY a1, paper.year
WITH a1, a2, collect(paper)[0].year AS year, count(*) AS collaborations
WHERE year < 2006
MERGE (a1)-[coauthor:CO_AUTHOR_EARLY {year: year}]-(a2)
SET coauthor.collaborations = collaborations;
// end::co-author-early[]

// tag::co-author-late[]
MATCH (a1)<-[:AUTHOR]-(paper)-[:AUTHOR]->(a2:Author)
WITH a1, a2, paper
ORDER BY a1, paper.year
WITH a1, a2, collect(paper)[0].year AS year, count(*) AS collaborations
WHERE year >= 2006
MERGE (a1)-[coauthor:CO_AUTHOR_LATE {year: year}]-(a2)
SET coauthor.collaborations = collaborations;
// end::co-author-late[]


// tag::co-author-apoc[]
CALL apoc.periodic.iterate(
 'MATCH (a1:Author)
  RETURN a1',
 'MATCH (a1)<-[:AUTHOR]-(paper)-[:AUTHOR]->(a2:Author)
  WITH a1, a2, paper
  ORDER BY a1, paper.year
  WITH a1, a2, collect(paper)[0].year AS year, count(*) AS collaborations
  MERGE (a1)-[coauthor:CO_AUTHOR {year: year}]-(a2)
  SET coauthor.collaborations = collaborations',
  {});
// end::co-author-apoc[]

// tag::co-author-early-apoc[]
CALL apoc.periodic.iterate(
 'MATCH (a1:Author)
  RETURN a1',
 'MATCH (a1)<-[:AUTHOR]-(paper)-[:AUTHOR]->(a2:Author)
  WITH a1, a2, paper
  ORDER BY a1, paper.year
  WITH a1, a2, collect(paper)[0].year AS year, count(*) AS collaborations
  WHERE year < 2006
  MERGE (a1)-[coauthor:CO_AUTHOR_EARLY {year: year}]-(a2)
  SET coauthor.collaborations = collaborations',
  {});
// end::co-author-early-apoc[]

// tag::co-author-late-apoc[]
 CALL apoc.periodic.iterate(
  'MATCH (a1:Author)
   RETURN a1',
  'MATCH (a1)<-[:AUTHOR]-(paper)-[:AUTHOR]->(a2:Author)
   WITH a1, a2, paper
   ORDER BY a1, paper.year
   WITH a1, a2, collect(paper)[0].year AS year, count(*) AS collaborations
   WHERE year >= 2006
   MERGE (a1)-[coauthor:CO_AUTHOR_LATE {year: year}]-(a2)
   SET coauthor.collaborations = collaborations',
   {});
// end::co-author-late-apoc[]
