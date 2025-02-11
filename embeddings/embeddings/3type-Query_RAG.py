import faiss
import torch
import clip
import google.generativeai as genai
from PIL import Image
import os
from typing import List, Tuple
import base64
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import torch.nn.functional as F
import shutil  # for copying files
import uuid
import numpy as np
import logging
from absl import logging as absl_logging

# Initialize logging at the beginning of your program
logging.basicConfig(level=logging.ERROR)  # You can set to CRITICAL or WARNING if you want fewer logs
absl_logging.set_verbosity(absl_logging.ERROR)  # Suppresses non-error logs from absl


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load environment variables
load_dotenv()

# Set device and load CLIP model (using a separate variable name for clarity)
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# Configure Gemini API Key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Define image directory (this is where uploaded images will be stored)
image_dir = "../images"


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


# Get current image paths
image_paths = get_image_paths(image_dir)


# Function to extract image embeddings from the CLIP model for a list of images
def get_features_from_image_path(image_paths_local: List[str]):
    """Extracts image embeddings from the CLIP model and normalizes them."""
    images = [preprocess(Image.open(image_path).convert("RGB")) for image_path in image_paths_local]
    image_input = torch.stack(images)
    with torch.no_grad():
        image_features_local = clip_model.encode_image(image_input).float()
    # Normalize embeddings for cosine similarity with an inner-product index.
    image_features_local = F.normalize(image_features_local, p=2, dim=-1)
    return image_features_local


# Function to extract embedding for a single image
def get_image_embedding(image_path: str):
    """Extracts and normalizes an embedding from an image using CLIP."""
    image = preprocess(Image.open(image_path).convert("RGB"))
    image_input = torch.unsqueeze(image, 0)
    with torch.no_grad():
        embedding = clip_model.encode_image(image_input).float()
    return F.normalize(embedding, p=2, dim=-1)


# Function to preprocess text
def preprocess_text(text: str) -> str:
    """Converts text to lowercase and removes extra spaces."""
    return " ".join(text.lower().strip().split())


# Function to extract and normalize text embeddings using CLIP
def get_text_embedding(text: str) -> torch.Tensor:
    """
    Extracts and normalizes a text embedding using CLIP.
    Returns a normalized tensor of shape (1, D) suitable for FAISS inner-product search.
    """
    processed_text = preprocess_text(text)
    text_input = clip.tokenize([processed_text]).to(device)
    with torch.no_grad():
        text_embedding = clip_model.encode_text(text_input).float()
    return F.normalize(text_embedding, p=2, dim=-1)


MAX_QUERY_LENGTH = 512  # Adjust according to the actual context length limit


def expand_query_with_gemini(query_text: str) -> str:
    """Uses Gemini to expand the text query and return the expanded version."""
    # Truncate the query to fit the maximum allowed length
    if len(query_text) > MAX_QUERY_LENGTH:
        query_text = query_text[:MAX_QUERY_LENGTH]
        print(f"Query truncated to {MAX_QUERY_LENGTH} characters.")

    chat_session = genai.GenerativeModel(
        model_name="gemini-2.0-flash-exp",
        generation_config={
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 150,
            "response_mime_type": "text/plain",
        },
    ).start_chat(history=[])

    response = chat_session.send_message([query_text, "Please make this query clear, concise and short."])
    expanded_query = response.text
    return expanded_query


# Build FAISS index using the current image embeddings
image_features = get_features_from_image_path(image_paths)
index = faiss.IndexFlatIP(image_features.shape[1])
index.add(image_features.numpy())


# Function to search for similar images using FAISS
def search_similar_images(index, query_embedding: torch.Tensor, top_k: int = 5):
    """Performs a FAISS search to find similar images."""
    query_embedding_np = query_embedding.unsqueeze(0).numpy()
    distances, indices = index.search(query_embedding_np, top_k)
    return [(os.path.basename(image_paths[i]), i, distances[0][j]) for j, i in enumerate(indices[0])]


# Function to resize an image and store it in a 'resized' subfolder
def resize_image(image_path: str, max_size: Tuple[int, int] = (1024, 1024)) -> str:
    """
    Resizes an image while maintaining aspect ratio, saving resized images in a separate folder.
    If a resized version already exists in that folder, it returns that file.
    """
    orig_dir = os.path.dirname(image_path)
    resized_dir = os.path.join(orig_dir, "resized")
    if not os.path.exists(resized_dir):
        os.makedirs(resized_dir)
    orig_filename = os.path.basename(image_path)
    name, _ = os.path.splitext(orig_filename)
    resized_filename = f"{name}_resized.jpg"
    resized_path = os.path.join(resized_dir, resized_filename)
    if os.path.exists(resized_path):
        return resized_path
    img = Image.open(image_path)
    img = img.convert("RGB")
    img.thumbnail(max_size)
    img.save(resized_path, "JPEG", quality=85)
    return resized_path


# Function to encode an image to base64
def encode_image(image_path: str) -> str:
    """Encodes an image to base64 format."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Function to upload an image to Gemini
def upload_to_gemini(image_path: str, mime_type: str = "image/jpeg"):
    """Resizes and uploads an image to Gemini, returning the file object."""
    resized_path = resize_image(image_path)
    file = genai.upload_file(resized_path, mime_type=mime_type)
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file


# Initialize the Gemini model (for image descriptions)
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


# Function to get a Gemini description for an image
def describe_image_with_gemini(image_path: str) -> str:
    """Uploads an image to Gemini and returns its description."""
    file = upload_to_gemini(image_path)
    chat_session = gemini_model.start_chat(history=[])
    response = chat_session.send_message([file, "Please describe this image and write 10 tag words"])
    return response.text


# Hybrid Search: Weighted combination of text-based and image-based retrieval
def hybrid_search(query_text: str, query_image_path: str, alpha: float, top_k: int = 5) -> List[Tuple[str, str, float]]:
    """
    Performs a hybrid search using both text and image queries.

    Args:
        query_text: The text query.
        query_image_path: The path to an image query.
        alpha: Weight for text query similarity (0 <= alpha <= 1). A value of 1 means only text, 0 means only image.
        top_k: Number of top results to return.

    Returns:
        A list of tuples: (image basename, full image path, combined similarity score).
    """
    # Get embeddings from both queries
    T = get_text_embedding(query_text)  # shape (1, d)
    I = get_image_embedding(query_image_path)  # shape (1, d)

    # Compute similarities with all stored image embeddings (assumed normalized)
    S_text = torch.mm(T, image_features.t())  # shape (1, N)
    S_image = torch.mm(I, image_features.t())  # shape (1, N)

    # Combine the similarity scores with weight alpha
    S_combined = alpha * S_text + (1 - alpha) * S_image  # shape (1, N)
    S_combined_np = S_combined.squeeze(0).cpu().numpy()  # shape (N,)

    # Get top_k indices sorted by combined similarity (higher is better)
    top_indices = S_combined_np.argsort()[::-1][:top_k]
    results = [(os.path.basename(image_paths[i]), image_paths[i], S_combined_np[i]) for i in top_indices]
    return results


# Function to upload a new image and update the FAISS index and image paths
def upload_new_image():
    """Uploads a new image to the image directory and updates the FAISS index and image_paths list."""
    new_image_source = input("Enter the path to the image you want to upload: ").strip()
    if not os.path.exists(new_image_source):
        print("The provided image does not exist.")
        return
    new_image_name = os.path.basename(new_image_source)
    new_image_destination = os.path.join(image_dir, new_image_name)
    if os.path.exists(new_image_destination):
        print("Image already exists in the directory. Skipping copy.")
    else:
        shutil.copy(new_image_source, new_image_destination)
        print(f"Image copied to {new_image_destination}")
    image_paths.append(new_image_destination)
    new_embedding = get_image_embedding(new_image_destination)
    if new_embedding is not None:
        new_embedding_np = new_embedding.numpy()  # new_embedding has shape (1, d)
        new_embedding_np = new_embedding_np.reshape(1, -1).astype('float32')
        index.add(new_embedding_np)
        print("FAISS index updated with new image.")
    else:
        print("Failed to compute embedding for the new image.")
    description = describe_image_with_gemini(new_image_destination)
    print(f"Gemini Description for new image: {description}")


# Main CLI: Choose Operation
print("Choose Operation:\n1. Query\n2. Upload\n3. Hybrid Query")
operation_choice = input("Enter your choice (1, 2, or 3): ").strip()

if operation_choice == "1":
    # Query Operation: Check if query is text or image based on extension
    query_input = input("Enter your query (text or an image filename): ").strip()
    if query_input.lower().endswith(('.jpg', '.jpeg', '.png')):
        query_image_path = os.path.join(image_dir, query_input)
        query_embedding = get_image_embedding(query_image_path)
    else:
        # Expand query using Gemini
        expanded_query = expand_query_with_gemini(query_input)
        print(f"Expanded Query: {expanded_query}")
        query_embedding = get_text_embedding(expanded_query)
    distances, indices = index.search(query_embedding.reshape(1, -1), 5)
    indices_distances = list(zip(indices[0], distances[0]))
    for idx, distance in indices_distances:
        print(f"Index: {idx}, Distance: {distance}")
        image_path = image_paths[idx]
        im = Image.open(image_path)
        plt.imshow(im)
        plt.axis('off')
        plt.show()
elif operation_choice == "2":
    upload_new_image()
elif operation_choice == "3":
    # Hybrid Query Operation
    query_text = input("Enter your text query: ").strip()
    expanded_query = expand_query_with_gemini(query_text)
    query_image_path = input("Enter the path to your image query: ").strip()
    query_image_path_hybrid = os.path.join(image_dir, query_image_path)
    alpha = float(input("Enter the alpha (weight) for text vs image similarity (0 <= alpha <= 1): ").strip())
    results = hybrid_search(expanded_query, query_image_path_hybrid, alpha)
    for res in results:
        print(f"Image: {res[0]}, Path: {res[1]}, Combined Similarity: {res[2]}")
        im = Image.open(res[1])
        plt.imshow(im)
        plt.axis('off')
        plt.show()
else:
    print("Invalid choice.")
