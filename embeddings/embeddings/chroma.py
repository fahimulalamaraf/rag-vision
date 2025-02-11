import chromadb
import os
import torch
import clip
import google.generativeai as genai
from PIL import Image
import base64
import matplotlib.pyplot as plt
import faiss
from dotenv import load_dotenv
from typing import List, Dict, Tuple
import numpy as np
import uuid

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load environment variables
load_dotenv()

# Set device and load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model.to(device)
for param in clip_model.parameters():
    param.data = param.data.contiguous()

# Configure Gemini API Key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize ChromaDB client and get/create the collection
chroma_client = chromadb.Client()
collection_name = "image_embeddings"
if collection_name in chroma_client.list_collections():
    collection = chroma_client.get_collection(name=collection_name)
else:
    collection = chroma_client.create_collection(name=collection_name)

# Image directory
image_dir = "../images"  # Replace with your image directory


# Function to get image paths
def get_image_paths(directory: str) -> List[str]:
    return [
        os.path.join(directory, filename)
        for filename in os.listdir(directory)
        if filename.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]


# Function to get image embedding
def get_image_embedding(image_path: str) -> torch.Tensor or None:
    try:
        image = preprocess(Image.open(image_path).convert("RGB"))
        image_input = torch.unsqueeze(image, 0).to(device)
        with torch.no_grad():
            embedding = clip_model.encode_image(image_input).float().cpu()
        return embedding
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None


# Improved Text Embedding Function (Normalize!)
def get_text_embedding(text: str) -> np.ndarray:
    """Extracts and normalizes embeddings from text using CLIP."""
    text_input = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_embedding = clip_model.encode_text(text_input).float().cpu().numpy()
    faiss.normalize_L2(text_embedding)  # Normalize text embedding
    return text_embedding


# Function to resize an image, using a cached version if available
def resize_image(image_path: str, max_size: Tuple[int, int] = (1024, 1024)) -> str:
    """Resizes image while maintaining aspect ratio.

    If a resized version already exists (or if the image_path already indicates a resized image),
    returns that file instead.
    """
    if image_path.endswith("_resized.jpg"):
        return image_path
    resized_path = f"{image_path}_resized.jpg"
    if os.path.exists(resized_path):
        return resized_path
    img = Image.open(image_path)
    img = img.convert("RGB")
    img.thumbnail(max_size)
    img.save(resized_path, "JPEG", quality=85)
    return resized_path


def encode_image(image_path: str) -> str:
    """Encodes an image to base64 format."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def upload_to_gemini(image_path: str, mime_type: str = "image/jpeg"):
    """Resizes and uploads an image to Gemini, returning the file object."""
    resized_path = resize_image(image_path)
    file = genai.upload_file(resized_path, mime_type=mime_type)
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file


# Example of a function that processes an image and returns metadata.
def process_image_and_metadata(image_path: str) -> Dict or None:
    """Processes an image to generate embedding, description, and tags."""
    embedding = get_image_embedding(image_path)
    if embedding is None:
        return None

    # Gemini API Call for Description and Tags
    try:
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
            encoded_image = base64.b64encode(image_bytes).decode("utf-8")
        gemini_model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
            generation_config={
                "temperature": 1,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 300,
                "response_mime_type": "text/plain",
            },
        )
        # Adjust model if needed
        response = gemini_model.generate_content(
            [
                {
                    "parts": [
                        {"inline_data": {"data": encoded_image, "mime_type": "image/jpeg"}},
                        {"text": "Describe this image and give me 10 single word tags."}
                    ]
                }
            ]
        )
        if response.candidates:
            gemini_output = response.candidates[0].content.parts[0].text
            try:
                description, tags_string = gemini_output.split("Tags:")
                description = description.strip()
                tags = [tag.strip() for tag in tags_string.split(",")]
            except ValueError:
                print(f"Unexpected Gemini output format: {gemini_output}")
                description = "Error getting description"
                tags = ["Error"]
        else:
            print("No response candidates from Gemini")
            description = "No description from Gemini"
            tags = ["No tags from Gemini"]
    except Exception as gemini_e:
        print(f"Error with Gemini API: {gemini_e}")
        description = f"Error with Gemini API: {gemini_e}"
        tags = ["Error"]

    metadata = {
        "description": description,
        "tags": tags,  # This is a list that we need to sanitize before adding to ChromaDB
        "image_path": image_path
    }
    document_content = f"{description} {' '.join(tags)}"
    return {"embedding": embedding.cpu().numpy(), "metadata": metadata, "document": document_content}


# Helper function to sanitize metadata values (convert lists to comma-separated strings)
def sanitize_metadata(metadata: Dict) -> Dict:
    sanitized = {}
    for key, value in metadata.items():
        if isinstance(value, list):
            sanitized[key] = ", ".join(map(str, value))
        else:
            sanitized[key] = value
    return sanitized


# Process images and build a list of records for ChromaDB
image_paths = get_image_paths(image_dir)
processed_images = []
for image_path in image_paths:
    data = process_image_and_metadata(image_path)
    if data:
        # Sanitize metadata here
        data['metadata'] = sanitize_metadata(data['metadata'])
        processed_images.append(data)

print(f"Processed {len(processed_images)} images.")

# Add records to ChromaDB in batches
batch_size = 100
for i in range(0, len(processed_images), batch_size):
    batch = processed_images[i:i + batch_size]
    embeddings = [item['embedding'] for item in batch]
    metadatas = [item['metadata'] for item in batch]
    documents = [item['document'] for item in batch]
    ids = [str(uuid.uuid4()) for _ in range(len(batch))]

    # Flatten embeddings and convert to lists
    embeddings_list = [emb.flatten().tolist() for emb in embeddings]

    collection.add(
        embeddings=embeddings_list,
        metadatas=metadatas,
        documents=documents,
        ids=ids
    )

print(f"Added {len(processed_images)} images to ChromaDB.")

# --- Example Query (after adding images) ---
query_text = "A cat on a mat"  # Example text query
query_embedding = get_text_embedding(query_text)

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5,
    where_document={"$contains": query_text},
)

if results and results.get('results') and results['results'][0]:
    for i, result in enumerate(results['results'][0]):
        image_path = results['metadatas'][0][i]['image_path']
        distance = result.get('distance', None)
        metadata = results['metadatas'][0][i]
        print(f"Image: {image_path}, Distance: {distance}")
        print(f"Metadata: {metadata}")
        try:
            im = Image.open(image_path)
            plt.imshow(im)
            plt.axis('off')
            plt.show()
        except Exception as e:
            print(f"Error displaying image: {e}")
else:
    print("No results found.")