from google import genai
from google.genai import types

client = genai.Client(api_key="AIzaSyBaL5sd4xDp_oStiGbQd_ZXNJbGfttsC50")

result = client.models.embed_content(
        model="text-embedding-004",
        contents="My name is Fahimul Alam")

print(result.embeddings)