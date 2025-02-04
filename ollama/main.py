import requests
import chromadb
import pprint


with open('../weathuringheights.txt', 'r', encoding='utf-8') as file:
    weathuringheights = file.read()

with open('../sonsoffire.txt', 'r', encoding='utf-8') as file:
    sonsoffire = file.read()

with open('../nautreofcrime.txt', 'r', encoding='utf-8') as file:
    natureofcrime = file.read()
#
# Initialize Chroma DB client
client = chromadb.Client()

database = client.create_collection("demo_A_db")

database.add(
    ids=["doc1", "doc2", "doc3"],
    metadatas=[{"topic":"sacrifice"}, {"topic":"Fire"}, {"topic":"Crime"}],
    documents=[ weathuringheights,
                sonsoffire,
                natureofcrime
                ]
)

# Query the vector database

query  = input("Enter a query: ")

query_text = query
results = database.query(
    query_texts=[query_text],
    n_results=1
)


if results["documents"]:
    document = results["documents"][0][0]

    # Define the Ollama API endpoint
    url = "http://localhost:11434//api/generate"


    prompt = f"Summarize the book and mention the document name:\n{document}"

    model = "llama3.2:1b"

    # Define the payload for the Ollama API
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    # Make the POST request to Ollama
    response = requests.post(url, json=payload)

    # Check the response
    if response.status_code == 200:
        print("Summary from Llama Model:")

        print(response.json().get("response", "No response found"))
        # print(response.json())
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
else:
    print("No documents found in the vector database.")
