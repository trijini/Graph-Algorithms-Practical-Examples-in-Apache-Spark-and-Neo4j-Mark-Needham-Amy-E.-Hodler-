// tag::price-range[]
MATCH (cat:Category)<-[:IN_CATEGORY]-(business:Business)-[:IN_CITY]->(city:City)
WHERE cat.name = $category AND city.name = $city AND not(exists(business.priceRange))
MATCH (business)<-[:REVIEWS]-()<-[:WROTE]-()-[:WROTE]->()-[:REVIEWS]->(other)
WHERE exists(other.priceRange)
WITH business, apoc.coll.avg(collect(other.priceRange)) AS priceRange
SET business.priceRange = toInteger(apoc.math.round(priceRange)),
    business.priceRangeEstimated = true
// end::price-range[]

// tag::price-range-missing[]
MATCH (cat:Category)<-[:IN_CATEGORY]-(business:Business)-[:IN_CITY]->(city:City)
WHERE cat.name = $category AND city.name = $city AND exists(business.priceRange)

WITH toInteger(apoc.math.round(avg(business.priceRange))) AS averagePriceRange
MATCH (cat:Category)<-[:IN_CATEGORY]-(business:Business)-[:IN_CITY]->(city:City)
WHERE cat.name = $category AND city.name = $city AND not(exists(business.priceRange))
SET business.priceRange = averagePriceRange,
    business.priceRangeEstimated = true
// end::price-range-missing[]


// tag::page-rank[]
CALL algo.pageRank(
  'MATCH (u:User) RETURN id(u) AS id',
  'MATCH (u1:User)-[:WROTE]->()-[:REVIEWS]->()-[:IN_CATEGORY]->(:Category {name: $category})
   MATCH (u1)-[:FRIENDS]->(u2)
   RETURN id(u1) AS source, id(u2) AS target',
   { graph: "cypher",
     write: true,
     writeProperty: "restaurantsPageRank",
     params: {category: "Restaurants"}
   })
// end::page-rank[]

// tag::feature-matrix-preprocessing[]
CALL apoc.periodic.iterate(
  'MATCH (user:User) RETURN user',
  'SET user.aveStars = apoc.coll.avg([(user)-[:WROTE]->(r) | r.stars])',
  {}
);
// end::feature-matrix-preprocessing[]

// tag::influential-friends[]
CALL apoc.periodic.iterate(
 'MATCH (user:User) RETURN user',
 'MATCH (user)-[:FRIENDS]->(other)
  WHERE other.restaurantsPageRank > 1.5
  WITH user, other
  ORDER BY user, other.restaurantsPageRank DESC
  WITH user, collect(other)[..5] AS influentialFriends
  UNWIND influentialFriends AS influentialFriend
  MERGE (user)-[friendship:INFLUENTIAL_FRIEND]->(influentialFriend)
  SET friendship.rank = influentialFriend.restaurantsPageRank',
  {});
// end::influential-friends[]

// tag::restaurant-attributes[]
call apoc.load.json("file:///home/markhneedham/projects/yelp-graph-algorithms/dataset-round12/yelp_academic_dataset_business.json")
YIELD value
WITH value WHERE value.categories contains "Restaurants"
MATCH (business:Business {id: value.business_id})
WITH business, value.attributes AS attrs
SET business.delivery = toBoolean(attrs.RestaurantsDelivery),
    business.goodForGroups = toBoolean(attrs.RestaurantsGoodForGroups),
    business.priceRange = toInteger(attrs.RestaurantsPriceRange2),
    business.attire = attrs.RestaurantsAttire,
    business.takeOut = toBoolean(attrs.RestaurantsTakeOut),
    business.reservations = toBoolean(attrs.RestaurantsReservations)
// end::restaurant-attributes[]

// tag::influential-friend-query[]
MATCH (cat:Category)<-[:IN_CATEGORY]-(business:Business)-[:IN_CITY]->(city:City)
WHERE cat.name = $category AND city.name = $city
MATCH (business:Business)<-[:REVIEWS]-(review:Review)<-[:WROTE]-(user:User)
RETURN review.id AS reviewId,
       business.id AS businessId,
       user.id AS userId,
       apoc.coll.avg([
         (business)<-[:REVIEWS]-(otherRev)<-[:WROTE]-(other:User)-[:INFLUENTIAL_FRIEND]-(user)
       	 WHERE otherRev.date < review.date | otherRev.stars]) AS aveOtherStars;
// end::influential-friend-query[]

// tag::find-similar-users[]
CALL algo.labelPropagation(
  'MATCH (u:LasVegas) WITH u SKIP {skip} LIMIT {limit} RETURN id(u) AS id',
  'MATCH (u1:LasVegas)
   WITH u1 SKIP {skip} LIMIT {limit}
   MATCH (u1)-[:WROTE]->()-[:REVIEWS]->(business)-[:IN_CATEGORY]->(:Category {name: $category}),
         (business)-[:IN_CITY]->(:City {name: $city}),
         (u2)-[:WROTE]->()-[:REVIEWS]->(business)
   RETURN id(u1) AS source, id(u2) AS target', 'OUTGOING',
   { graph: "cypher",
     params: {category: "Restaurants", city: "Las Vegas"}
   })
// end::find-similar-users[]


CALL algo.graph.load('similar-users',
 'MATCH (u:LasVegas) WITH u SKIP {skip} LIMIT {limit} RETURN id(u) AS id',
 'MATCH (u1:LasVegas)
  WITH u1 SKIP {skip} LIMIT {limit}
  MATCH (u1)-[:WROTE]->()-[:REVIEWS]->(business)-[:IN_CATEGORY]->(:Category {name: "Restaurants"}),
        (business)-[:IN_CITY]->(:City {name: "Las Vegas"}),
        (u2)-[:WROTE]->()-[:REVIEWS]->(business)
  RETURN id(u1) AS source, id(u2) AS target',
 {graph:'cypher'})

 CALL algo.graph.load('similar-users-weight',
 'MATCH (u:LasVegas) WITH u SKIP {skip} LIMIT {limit} RETURN id(u) AS id',
 'MATCH (u1:LasVegas)
  WITH u1 SKIP {skip} LIMIT {limit}
  MATCH (u1)-[:WROTE]->()-[:REVIEWS]->(business)-[:IN_CATEGORY]->(:Category {name: "Restaurants"}),
        (business)-[:IN_CITY]->(:City {name: "Las Vegas"}),
        (u2)-[:WROTE]->()-[:REVIEWS]->(business)
  RETURN id(u1) AS source, id(u2) AS target, 1 as weight',
 {graph:'cypher'})

 CALL algo.labelPropagation(null, null, "OUTGOING", { graph: "similar-users-weight", partitionProperty: "foo" })
