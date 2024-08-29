# ScribbleAI: Personalized Blog Writing Companion
ScribbleAI is a GenAI-powered tool designed to help users create personalized blog posts from their ideas. By leveraging the power of the Gemini API and Streamlit, ScribbleAI simplifies the content creation process, making writing faster and more efficient. Whether you're looking to enhance your creativity or overcome writer's block, ScribbleAI is here to assist.

# Features
Personalized Blog Generation: Generate blog posts based on your unique ideas with the help of AI.
User-Friendly Interface: Built with Streamlit, the interface is intuitive and easy to use.
Efficient Content Creation: ScribbleAI accelerates the writing process, making it ideal for both casual bloggers and professionals.
Creativity Enhancement: Get inspiration and overcome writer's block by letting AI assist with your blog writing.
# How It Works
Input Your Idea: Enter the topic or main idea for your blog post.
Generate Content: ScribbleAI uses the Gemini API to create a personalized blog post based on your input.
Edit & Publish: Review the generated content, make any necessary edits, and your blog post is ready to be published!
## Tech Stack

- **Frontend**: Streamlit
- **Backend**: Gemini API
- **API's Used**: Google Gemini 1.5 Pro and Hugging Face Stable diffusion v.1.5
- **Programming Language**: Python

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/Murghendra/ScribbleAI.git
    cd ScribbleAI
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up your environment variables:
    - Create a `.env` file in the root directory and add your API keys:
      ```bash
      GEMINI_API_KEY=your_gemini_api_key
      Hugging_Face_API_Key=hugging_face_key
      ```

4. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

# Screenshots
![Snapshot](/images/1.png)

