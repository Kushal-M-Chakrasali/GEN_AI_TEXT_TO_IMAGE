import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

st.set_page_config(page_title="KMC AI Text to Image", layout="centered")

# Load Logo
logo = Image.open("KMC_Transperent.jpeg")
st.image(logo, width=200)

st.title("KMC AI - Text to Image Generator")

prompt = st.text_input("Enter your prompt")

@st.cache_resource
def load_model():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    )
    pipe = pipe.to("cpu")
    return pipe

if st.button("Generate Image"):
    if prompt != "":
        with st.spinner("Generating Image..."):
            pipe = load_model()
            image = pipe(prompt).images[0]

            st.image(image, caption="Generated Image", use_container_width=True)

            image.save("generated_image.png")

            with open("generated_image.png", "rb") as file:
                st.download_button(
                    label="Download Image",
                    data=file,
                    file_name="generated_image.png",
                    mime="image/png"
                )
    else:
        st.warning("Please enter a prompt")
