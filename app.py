import os
from dotenv import load_dotenv
import time
import streamlit as st
import google.generativeai as genai
from openai import OpenAI
import requests
import io
from PIL import Image, UnidentifiedImageError

# Retrieve API keys from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
google_gemini_api_key = os.getenv("GOOGLE_GEMINI_API_KEY")
hugging_face_api_key = os.getenv("HUGGING_FACE_API_KEY")

client = OpenAI(api_key=openai_api_key)

# Configure the API key for Google Gemini
genai.configure(api_key=google_gemini_api_key)

# Create the model configuration for text generation
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Set up the Gemini model
model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config=generation_config,
)

# Set layout as wide
st.set_page_config(layout="wide")

# Title of the app
st.title('ScribbleAI✍️: Personalized Blog Writing Companion')

# Subheader
st.subheader('"Unleash your creativity with AI-powered writing ✍️, tailored to your unique keywords 🔑."')

# Sidebar for user inputs
with st.sidebar:
    st.image("blog.jpg", use_column_width=True)  # Replace with the path to your JPG image    
    st.title("Input Your Blog Details")
    st.subheader("Enter details of the Blog you want to generate")
    
    # Blog title
    blog_title = st.text_input("Blog Title")
    
    # Keywords input by user
    keywords = st.text_area("Keywords (separated by comma)")
    
    # Number of words
    num_words = st.slider("Number of Words", min_value=100, max_value=1000, step=100)
    
    # Number of images
    num_images = st.number_input("Number of Images", min_value=1, max_value=5, step=1)
    
    # Button to generate blog
    submit_button = st.button("Generate Blog")

if submit_button:
    # Generate multiple images based on blog title using Hugging Face Stable Diffusion
    API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
    headers = {"Authorization": f"Bearer {hugging_face_api_key}"}
    
    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response

    # Container for images
    image_container = st.empty()

    for i in range(num_images):
        image_prompt = f"Generate an image that visually represents the title: '{blog_title}'."
        
        max_retries = 5
        for j in range(max_retries):
            image_response = query({"inputs": image_prompt})
            
            if image_response.status_code == 503:
                st.write(f"Model is still loading. Retrying in 30 seconds... ({j+1}/{max_retries})")
                time.sleep(30)  # Wait for 30 seconds before retrying
            elif image_response.status_code == 500:
                st.write("Server encountered an error. Retrying in 60 seconds...")
                time.sleep(60)  # Wait for 60 seconds before retrying
            else:
                break
        
        # Debugging: Check the response content type
        st.write(f"Image generation attempt {i+1}:")
        st.write("Response status code:", image_response.status_code)
        st.write("Response content type:", image_response.headers.get('Content-Type'))

        try:
            image = Image.open(io.BytesIO(image_response.content))
            # Display images
            image_container.image(image, caption=f"Generated image {i+1} for: {blog_title}")
        except UnidentifiedImageError:
            st.write(f"Failed to generate image {i+1}. The API did not return a valid image.")
            st.write("Response content:", image_response.content.decode())
    
    # Generate blog content below images
    prompt = f"Generate a comprehensive, engaging blog post relevant to the given title \"{blog_title}\" and keywords \"{keywords}\". Make sure to incorporate these keywords in the blog post. The blog should be approximately {num_words} words in length, suitable for an online audience. Ensure the content is original, informative, and maintains a consistent tone throughout."

    # Start chat session with Gemini
    chat_session = model.start_chat(
        history=[
            {
                "role": "user",
                "parts": [prompt]
            }
        ]
    )

    # Send message and get response from Gemini
    response = chat_session.send_message(
        {
            "role": "user",
            "parts": [prompt]
        }
    )
        
    # Display the generated blog content
    st.write(response.text)



    
    
