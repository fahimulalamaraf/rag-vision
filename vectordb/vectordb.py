import chromadb
import pprint

client = chromadb.Client()

database = client.create_collection("demo_A_db")

database.add(
    ids=["doc1", "doc2", "doc3"],
    metadatas=[{"topic":"food"}, {"topic":"animal-food"}, {"topic":"animal"}],
    documents=[
        "This is a document about food.",
        "This is a document about amimal's food.",
        "This is a document about dog and cat."
    ]
)

results = database.query(
    query_texts=["dog"],
    n_results=2
)

pprint.pprint(results)
