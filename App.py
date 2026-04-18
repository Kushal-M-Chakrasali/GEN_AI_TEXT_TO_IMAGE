import streamlit as st
import os
from huggingface_hub import InferenceClient
from PIL import Image

# Set page config
st.set_page_config(page_title="KMC AI Text to Image", layout="centered")

# Logo
logo = Image.open("KMC_Transperent.jpeg")
st.image(logo, width=250)

st.title("KMC AI - Text to Image Generator")

# Prompt input
prompt = st.text_input("Enter your prompt")

# Hugging Face token
HF_TOKEN = os.getenv("HF_TOKEN")

# Create client
client = InferenceClient(
    provider="fal-ai",
    api_key=HF_TOKEN,
)

if st.button("Generate Image"):

    if prompt.strip() == "":
        st.warning("Please enter a prompt")

    else:
        with st.spinner("Generating image..."):

            image = client.text_to_image(
                prompt,
                model="baidu/ERNIE-Image",
            )

            st.image(image, caption="Generated Image", use_container_width=True)

            # Save image
            image.save("generated_image.png")

            # Download button
            with open("generated_image.png", "rb") as file:
                st.download_button(
                    label="Download Image",
                    data=file,
                    file_name="generated_image.png",
                    mime="image/png"
                )
