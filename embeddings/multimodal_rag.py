import faiss
import os
import torch
import clip
import google.generativeai as genai
from PIL import Image
import base64
import matplotlib.pyplot as plt
from dotenv import load_dotenv
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load environment variables
load_dotenv()

# Set device and load CLIP model
device = "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# Configure Gemini API Key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


# # Vector DB Indexing with FAISSS
# index = faiss.IndexFlatIP(image_features.shape[1])
# index.add(image_features)
#
# # Save the db in disk
# faiss.write_index(index, "image_embeddings.index")
# Load FAISS index
index = faiss.read_index("image_embeddings.index")

# Set image directory
image_dir = '../images'


# Get image paths
def get_image_paths(directory: str):
    """Retrieve full paths of all images in a directory."""
    return [
        os.path.join(directory, filename)
        for filename in os.listdir(directory)
        if filename.lower().endswith(('.jpg' or '.jpeg' or '.png'))
    ]


image_paths = get_image_paths(image_dir)


# Extract image embeddings
def get_image_embedding(image_path: str):
    """Extracts embeddings from an image using CLIP."""
    image = preprocess(Image.open(image_path).convert("RGB"))
    image_input = torch.unsqueeze(image, 0)  # Add batch dimension
    with torch.no_grad():
        embedding = clip_model.encode_image(image_input).float()
    return embedding


# Extract text embeddings
def get_text_embedding(text: str):
    """Extracts embeddings from text using CLIP."""
    text_input = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_embedding = clip_model.encode_text(text_input).float()
    return text_embedding


# Search similar images in FAISS
def search_similar_images(query_embedding: torch.Tensor, top_k: int = 5):
    """Search for similar images in FAISS using the query embedding."""
    query_embedding = query_embedding.cpu().numpy()  # Convert to NumPy
    distances, indices = index.search(query_embedding, top_k)
    return [(image_paths[i], distances[0][j]) for j, i in enumerate(indices[0])]


# Resize and encode image for Gemini
def resize_image(image_path: str, max_size=(1024, 1024)) -> str:
    """Resizes image while maintaining aspect ratio and saves it."""
    resized_path = f"{image_path}_resized.jpg"
    if os.path.exists(resized_path):
        return resized_path

    img = Image.open(image_path)
    img = img.convert("RGB")
    img.thumbnail(max_size)
    img.save(resized_path, "JPEG", quality=85)
    return resized_path


# Upload image to Gemini
def upload_to_gemini(image_path: str):
    """Uploads an image to Gemini and returns the file object."""
    resized_path = resize_image(image_path)
    file = genai.upload_file(resized_path, mime_type="image/jpeg")
    return file


# Describe image with Gemini
def describe_image_with_gemini(image_path: str) -> str:
    """Uploads an image to Gemini and retrieves its description."""
    file = upload_to_gemini(image_path)
    gemini_model = genai.GenerativeModel("gemini-2.0-flash-exp")
    chat_session = gemini_model.start_chat(history=[])
    response = chat_session.send_message([file, "Describe this image and generate 10 relevant tags"])
    return response.text


# User Selection: Text Query or Image Query
print("Choose Query Type:\n1. Text Query\n2. Image Query")
choice = input("Enter your choice (1 or 2): ").strip()

# Perform search based on user selection
if choice == "1":
    # Text Query
    text_query = input("Enter your text query: ").strip()
    query_embedding = get_text_embedding(text_query)
elif choice == "2":
    # Image Query
    image_query_path = "../images/" + input("Enter image path: ")
    query_embedding = get_image_embedding(image_query_path)
else:
    print("Invalid choice. Exiting.")
    exit()

# Search for similar images
top_k = 4
similar_images = search_similar_images(query_embedding, top_k)

# Display results
for image_path, distance in similar_images:
    print(f"Found Image: {image_path}, Distance: {distance}")
    im = Image.open(image_path)
    plt.imshow(im)
    plt.axis('off')
    plt.show()

    # Get Gemini description
    # description = describe_image_with_gemini(image_path)
    # print(f"Gemini Description: {description}")
