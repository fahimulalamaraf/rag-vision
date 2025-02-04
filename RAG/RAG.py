from groq import Groq
import os
from dotenv import load_dotenv
import base64
import requests
import io
from PIL import Image



load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

client = Groq()

filepath = f"E:\Projects\Large-LM\images\plowers.jpg"
image = Image.open(filepath)

image_bytes = io.BytesIO()
image.save(image_bytes, format='JPEG')
image_bytes = image_bytes.getvalue()

# Step 3: Encode the bytes into Base64 string
image_base64 = base64.b64encode(image_bytes).decode('utf-8')

# Download the image and convert to base64
# image_url = "E:\Projects\Large-LM\images\jackie-best-5UKeDgV6PNg-unsplash.jpg"
# response = requests.get(image_url)
# image_base64 = base64.b64encode(response.content).decode("utf-8")



# Send Base64 Image instead of URL
completion = client.chat.completions.create(
    model="llama-3.2-90b-vision-preview",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Write description of the image in two sentences. Generate tags of the images."},
                # {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
            ]
        }
    ],
    temperature=1,
    max_completion_tokens=1024
)

print(completion.choices[0].message.content)
