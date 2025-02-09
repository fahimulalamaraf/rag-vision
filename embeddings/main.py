import faiss
import json
import torch
from openai import OpenAI
import torch.nn as nn
from torch.utils.data import DataLoader
import clip
client = OpenAI()
import google.generativeai as genai

# helper imports
from tqdm import tqdm
import json
import os
import numpy as np
import pickle
from typing import List, Union, Tuple
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# visualisation imports
from PIL import Image
import matplotlib.pyplot as plt
import base64
from dotenv import load_dotenv

load_dotenv()

device = "cpu"
model, preprocess = clip.load("ViT-B/32",device=device)


def get_image_paths(directory: str, number: int = None) -> List[str]:

    """Retrieves the file paths of all JPG, JPEG, PNG images in a given directory.

    This function scans the specified directory for images with `.jpg` or `.jpeg`
    extensions and returns a list of their full file paths. Optionally, a limit can be
    set to retrieve only the first `number` images.

    Args:
        directory (str): The directory containing image files.
        number (int, optional): The number of image paths to return.
            If None, returns all images. Defaults to None.

    Returns:
        List[str]: A list of full file paths of the images.

    Raises:
        FileNotFoundError: If the specified directory does not exist.

    Example:
        >>> get_image_paths("image_database", number=3)
        ["image_database/image1.jpg", "image_database/image2.jpg", "image_database/image3.jpg"]
    """

    image_paths_local = []
    count = 0
    for filename in os.listdir(directory):
        if filename.lower().endswith('.jpg' or '.jpeg' or '.png'):
            image_paths_local.append(os.path.join(directory, filename))
            if number is not None and count == number:
                return image_paths_local[:number]
            count += 1
    return image_paths_local
direc = '../images'
image_paths = get_image_paths(direc)

# print(len(image_paths))


def get_features_from_image_path(image_paths_local):

  images = [preprocess(Image.open(image_path).convert("RGB")) for image_path in image_paths_local]
  image_input = torch.stack(images)
  with torch.no_grad():
    image_features_local = model.encode_image(image_input).float()
  return image_features_local
image_features = get_features_from_image_path(image_paths)

# print(image_features)

# Vector DB Indexing with FAISSS
index = faiss.IndexFlatIP(image_features.shape[1])
index.add(image_features)

# Save the db in disk
faiss.write_index(index, "image_embeddings.index")

#Load the db in disk
# index = faiss.read_index("image_embeddings.index")


# def search_similar_images(index, query_embedding: torch.Tensor, top_k: int = 5):
#     """Finds the top-k most similar images to a given query embedding.
#
#     Args:
#         index (faiss.IndexFlatIP): The FAISS index containing image embeddings.
#         query_embedding (torch.Tensor): A single image embedding to search for.
#         top_k (int): The number of similar images to retrieve.
#
#     Returns:
#         List[int]: The indices of the top-k most similar images.
#
#     Example:
#         >>> query_embedding = get_features_from_image_path(["query.jpg"])
#         >>> similar_indices = search_similar_images(index, query_embedding)
#         >>> print(similar_indices)
#         [5, 12, 9, 34, 21]  # Indices of similar images
#     """
#     query_embedding = query_embedding.unsqueeze(0).numpy()
#     distances, indices = index.search(query_embedding, top_k)  # FAISS search
#     return [(os.path.basename(image_paths[i]), i) for i in indices[0]]  # Convert NumPy array to Python list
#
# # Example search (assumes query image is preprocessed)
# # query_image_path = "../images/marina-reich-yrgtdhqJZck-unsplash.jpg"
# # query_embedding = get_features_from_image_path([query_image_path])
# # similar_images = search_similar_images(index, query_embedding)
# # print(f"Similar image indices: {similar_images}")
#
#
# def resize_image(image_path: str, max_size: Tuple[int, int] = (1024, 1024)) -> str:
#     """Resizes an image while maintaining aspect ratio and saves it as a temporary file.
#
#     Args:
#         image_path (str): Path to the original image.
#         max_size (Tuple[int, int]): Maximum width & height (default: 1024x1024).
#
#     Returns:
#         str: Path to the resized image.
#     """
#     img = Image.open(image_path)
#
#     # Convert to RGB to avoid mode issues (e.g., PNG transparency)
#     img = img.convert("RGB")
#
#     # Resize while keeping aspect ratio
#     img.thumbnail(max_size)  # ✅ Fixed: Removed Image.ANTIALIAS
#
#     # Save resized image as a temporary file
#     resized_path = f"{image_path}_resized.jpg"
#     img.save(resized_path, "JPEG", quality=85)  # Adjust quality for compression
#
#     return resized_path
#
#
#
# # # Configure Gemini API Key
# # genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# #
# # def encode_image(image_path: str) -> str:
# #     """Encodes an image to base64 format."""
# #     with open(image_path, "rb") as image_file:
# #         return base64.b64encode(image_file.read()).decode("utf-8")
# #
# # def upload_to_gemini(image_path: str, mime_type: str = "image/jpeg"):
# #     """Uploads an image to Gemini and returns a file object."""
# #     resized_path = resize_image(image_path)
# #     file = genai.upload_file(resized_path, mime_type=mime_type)
# #     print(f"Uploaded file '{file.display_name}' as: {file.uri}")
# #     return file
# #
# # # Gemini model configuration
# # generation_config = {
# #     "temperature": 1,
# #     "top_p": 0.95,
# #     "top_k": 40,
# #     "max_output_tokens": 1000,  # Adjust output length as needed
# #     "response_mime_type": "text/plain",
# # }
# #
# # # Initialize the Gemini model
# # model = genai.GenerativeModel(
# #     model_name="gemini-2.0-flash-exp",
# #     generation_config=generation_config,
# # )
# #
# # def image_query(query: str, image_path: str) -> str:
# #     """Queries Gemini with an image and text prompt.
# #
# #     Args:
# #         query (str): The textual prompt for the vision model.
# #         image_path (str): Path to the image file.
# #
# #     Returns:
# #         str: Model's textual response describing the image.
# #     """
# #     # Upload image to Gemini
# #     file = upload_to_gemini(image_path, mime_type="image/jpeg")
# #
# #     # Start a chat session with the uploaded image
# #     chat_session = model.start_chat(history=[])
# #
# #     response = chat_session.send_message([
# #         file,  # Attach the image
# #         query  # Add user query
# #     ])
# #
# #     return response.text
#
# # ✅ Example Usage
# query_image_path = "../images/natalia-grela-Wdz8MQ_eSA4-unsplash.jpg"
# # label = image_query("Write a short description of the image in "
# #                     "\ two sentences and Generate tags for the image.", query_image_path)
# # print(f"Gemini Label: {label}")
#
# # Get the embedding of the query image
# image_search_embedding = get_features_from_image_path([query_image_path])
#
# # Perform FAISS search for the top 2 similar images
# distances, indices = index.search(image_search_embedding.reshape(1, -1), 3) #3 is similarity match
#
# # Extract first results (already sorted by FAISS)
# indices_distances = list(zip(indices[0], distances[0]))
#
# # Display images based on the retrieved indices
# for idx, distance in indices_distances:
#     print(f"Index: {idx}, Distance: {distance}")
#     path = image_paths[idx]
#     im = Image.open(path)
#     plt.imshow(im)
#     plt.show()
#
