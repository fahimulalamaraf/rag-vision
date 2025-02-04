from groq import Groq
import os
from dotenv import load_dotenv
import base64
import requests
import io
from PIL import Image
from pathlib import Path

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

client = Groq()

filepath = Path("D:/Projects/LLM with Images/rag-vision/images/plowers.jpg")
image = Image.open(filepath)

image_bytes = io.BytesIO()
image.save(image_bytes, format='JPEG')
image_bytes = image_bytes.getvalue()

# Step 3: Encode the bytes into Base64 string
image_base64 = base64.b64encode(image_bytes).decode('utf-8')



# Send Base64 Image instead of URL
completion = client.chat.completions.create(
    model="llama-3.2-90b-vision-preview",
    messages=[
        {
            # Define the system role (assistant role/purpose)
            "role": "system",
            "content": "You are an AI assistant capable of analyzing images. Respond with descriptions and helpful information."
        },
        {
            # Assistant's friendly introduction
            "role": "assistant",
            "content": "Well hello there! I am Pexels. I will assist you in describing your images with precision!"
        },
        {
            # User's query with image and task input
            "role": "user",
            "content": [
                {"type": "text",
                 "text": "Write description of the image in two sentences. Generate tags of the images."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
            ]
        }
    ],
    temperature=1,
    max_completion_tokens=1024
)

# Display the response
print(completion)
