from py2neo import Graph,NodeMatcher


graph = Graph("localhost:7474", username="neo4j", password="1039")
matcher = NodeMatcher(graph)
ent = "晕厥"
pro = "常见病因"

print(matcher.match("Qands", Qid=1).first())
