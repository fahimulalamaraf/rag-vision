import faiss
import json
import torch
import clip
import google.generativeai as genai
from PIL import Image
import os
from typing import List, Union, Tuple
import base64
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set device and load CLIP model (using a separate variable name for clarity)
device = "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# Configure Gemini API Key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


def get_image_paths(directory: str, number: int = None) -> List[str]:
    """Retrieves full file paths of images in a directory."""
    image_paths_local = []
    count = 0
    # Use a tuple for multiple extensions
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_paths_local.append(os.path.join(directory, filename))
            if number is not None and count == number:
                return image_paths_local[:number]
            count += 1
    return image_paths_local


# Set image directory and get image paths
direc = '../images'
image_paths = get_image_paths(direc)


def get_features_from_image_path(image_paths_local: List[str]):
    """Extracts image embeddings from the CLIP model."""
    images = [preprocess(Image.open(image_path).convert("RGB")) for image_path in image_paths_local]
    image_input = torch.stack(images)
    with torch.no_grad():
        image_features_local = clip_model.encode_image(image_input).float()
    return image_features_local


# Extract features for all images in our knowledge base
image_features = get_features_from_image_path(image_paths)

# Load the FAISS index from disk (make sure the index was built with the same ordering as image_paths)
index = faiss.read_index("image_embeddings.index")


def search_similar_images(index, query_embedding: torch.Tensor, top_k: int = 5):
    """Performs a FAISS search to find similar images."""
    query_embedding = query_embedding.unsqueeze(0).numpy()
    distances, indices = index.search(query_embedding, top_k)
    return [(os.path.basename(image_paths[i]), i) for i in indices[0]]


def resize_image(image_path: str, max_size: Tuple[int, int] = (1024, 1024)) -> str:
    """Resizes an image while maintaining aspect ratio.

    If a resized version already exists (or if the given image_path already
    indicates a resized image), it returns that file instead.

    Args:
        image_path (str): Path to the original image.
        max_size (Tuple[int, int], optional): Maximum dimensions (width, height). Defaults to (1024, 1024).

    Returns:
        str: Path to the resized image.
    """
    # Check if the image path already indicates a resized version
    if image_path.endswith("_resized.jpg"):
        return image_path

    resized_path = f"{image_path}_resized.jpg"
    # Check if the resized file already exists on disk
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


# Initialize the Gemini model using a separate variable name
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


def describe_image_with_gemini(image_path: str) -> str:
    """Uploads an image to Gemini and returns its description."""
    file = upload_to_gemini(image_path)
    # Start a chat session with the uploaded image using the Gemini model
    chat_session = gemini_model.start_chat(history=[])
    response = chat_session.send_message([file, "Please describe this image and write 10 tag words"])
    return response.text


# Get the embedding of the query image using CLIP
query_image_path = "../images/" + input("Enter image path: ")
query_embedding = get_features_from_image_path([query_image_path])

# Perform FAISS search for the top 4 similar images
distances, indices = index.search(query_embedding.reshape(1, -1), 4)
indices_distances = list(zip(indices[0], distances[0]))

# Display images and request Gemini descriptions
for idx, distance in indices_distances:
    print(f"Index: {idx}, Distance: {distance}")
    image_path = image_paths[idx]
    im = Image.open(image_path)
    plt.imshow(im)
    plt.axis('off')
    plt.show()
    description = describe_image_with_gemini(image_path)
    print(f"Description from Gemini: {description}")
