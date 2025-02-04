import os
from dotenv import load_dotenv
import google.generativeai as genai
from pathlib import Path

# Load the environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Configure the Generative AI client
genai.configure(api_key=api_key)


def upload_to_gemini(path, mime_type=None):
    """Uploads the given file to Gemini.

    See https://ai.google.dev/gemini-api/docs/prompting_with_media
    """
    file = genai.upload_file(path, mime_type=mime_type)
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file


# Create the model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
    system_instruction="You are a photo gallery AI assistant named Pixels. You can retrieve images as per the user's query from the user's gallery and describe them. Users can ask you for memories from you and you will present them in front of the user.",
)

# File path to the local image
image_path = "D:/Projects/LLM with Images/rag-vision/images/jackie-best-5UKeDgV6PNg-unsplash.jpg"

# Upload the image file to Gemini and get its URI
uploaded_file = upload_to_gemini(image_path, mime_type="image/jpeg")

# Define the chat session with updated history
chat_session = model.start_chat(
    history=[
        {
            "role": "user",
            "parts": ["Hello. Who are you?\n"],
        },
        {
            "role": "model",
            "parts": [
                "Hello! I'm Pixels, your friendly AI photo gallery assistant. I can help you find and relive your memories by searching your photo gallery and describing the images you're looking for. Just let me know what you're looking for!\n",
            ],
        },
        {
            "role": "user",
            "parts": [
                {  # Properly passing the uploaded image's URI in the expected format
                    "mime_type": "image/jpeg",
                    "data": uploaded_file.uri,  # Pass the uploaded file's URI
                },
                "Explain the image in two sentences and find tags.\n",
            ],
        },
    ]
)

# Send a message to the chat session for describing the image
response = chat_session.send_message("Describe the uploaded image.")
print(response.text)  # Display the response from the AI model
