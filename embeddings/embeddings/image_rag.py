import os
import faiss
import torch
import clip
import numpy as np
from typing import List, Tuple
import logging
from dotenv import load_dotenv
from PIL import Image
import google.generativeai as genai
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL, PREPROCESS = clip.load("ViT-B/32", device=DEVICE)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
FAISS_INDEX_FILE = "faiss_index.bin"
FEATURES_FILE = "image_features.npy"
MAX_QUERY_LENGTH = 200
image_dir = "../images"

# Initialize Gemini model
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


def get_features_from_image_path(image_paths: list) -> torch.Tensor:
    """Extracts and returns image features from CLIP model."""
    features = []
    for path in image_paths:
        try:
            image = Image.open(path).convert("RGB")
            image = PREPROCESS(image).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                feature = MODEL.encode_image(image)
            features.append(feature.cpu())
        except Exception as e:
            logging.error(f"Error processing {path}: {e}")
    return torch.cat(features, dim=0) if features else torch.empty(0)


def save_faiss_index(index: faiss.IndexFlatIP, index_file: str = FAISS_INDEX_FILE):
    faiss.write_index(index, index_file)


def load_faiss_index(index_file: str = FAISS_INDEX_FILE, dim: int = 512) -> faiss.IndexFlatIP:
    if os.path.exists(index_file):
        return faiss.read_index(index_file)
    return faiss.IndexFlatIP(dim)


def save_features(features: torch.Tensor):
    np.save(FEATURES_FILE, features.numpy())


def load_features():
    return torch.tensor(np.load(FEATURES_FILE)) if os.path.exists(FEATURES_FILE) else None


def expand_query_with_gemini(query_text: str) -> str:
    """Expands user query using Gemini API."""
    prompt = f"Refine this search query concisely: {query_text}"
    chat_session = gemini_model.start_chat(history=[])
    response = chat_session.send_message([prompt])
    expanded_query = response.text.strip()
    return expanded_query[:MAX_QUERY_LENGTH] if expanded_query else query_text


def get_text_embedding(text: str) -> torch.Tensor:
    """Encodes text query into CLIP embeddings."""
    with torch.no_grad():
        text_features = MODEL.encode_text(clip.tokenize([text]).to(DEVICE))
    return text_features.cpu()


def describe_image_with_gemini(image_path: str) -> str:
    """Generates an image description using Gemini."""
    chat_session = gemini_model.start_chat(history=[])
    response = chat_session.send_message([f"Describe this image: {image_path}"])
    return response.text.strip() if response else ""


def hybrid_search_with_description(query_text: str, query_image_path: str) -> torch.Tensor:
    """Combines text and image descriptions for hybrid search."""
    try:
        image_description = describe_image_with_gemini(query_image_path) or "generic image"
    except Exception as e:
        logging.error(f"Error generating image description: {e}")
        image_description = "generic image"
    final_query = expand_query_with_gemini(f"{query_text} {image_description}")
    return get_text_embedding(final_query)

# Function to get image paths from a directory
def get_image_paths(directory: str, number: int = None) -> List[str]:
    """Retrieves full file paths of images in a directory."""
    image_paths_local = []
    count = 0
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_paths_local.append(os.path.join(directory, filename))
            if number is not None and count == number:
                return image_paths_local[:number]
            count += 1
    return image_paths_local


def main():
    """Main function to execute the search pipeline."""
    image_paths = get_image_paths(image_dir)

    # Load or compute image features
    image_features = load_features()
    if image_features is None:
        image_features = get_features_from_image_path(image_paths)
        save_features(image_features)

    # Load or create FAISS index
    index = load_faiss_index()
    if index.ntotal == 0:
        index.add(image_features.numpy())
        save_faiss_index(index)

    # User Query Input
    query_text = input("Enter search query: ")
    query_image_path = input("Enter image path for hybrid search (or press Enter to skip): ")

    query_embedding = hybrid_search_with_description(query_text,
                                                     query_image_path) if query_image_path else get_text_embedding(
        query_text)

    # FAISS Search
    D, I = index.search(query_embedding.numpy(), k=5)
    results = [image_paths[i] for i in I[0] if i >= 0]

    print("Top Matches:")
    for img in results:
        print(img)

if __name__ == "__main__":
    main()
