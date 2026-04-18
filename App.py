import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

# Page configuration
st.set_page_config(page_title="KMC AI Text to Image", layout="centered")

# Display Logo
logo = Image.open("KMC_Transperent.jpeg")
st.image(logo, width=250)

st.title("KMC AI - Text to Image Generator")

st.write("Generate images from text prompts using Stable Diffusion.")

# Prompt input
prompt = st.text_input("Enter your prompt")

# Load lightweight model
@st.cache_resource
def load_model():
    model_id = "stabilityai/sd-turbo"

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32
    )

    pipe = pipe.to("cpu")

    return pipe


pipe = load_model()

# Generate button
if st.button("Generate Image"):

    if prompt.strip() == "":
        st.warning("Please enter a prompt")

    else:
        with st.spinner("Generating image..."):
            image = pipe(prompt, num_inference_steps=1, guidance_scale=0.0).images[0]

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
