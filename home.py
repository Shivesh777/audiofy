import streamlit as st
from PIL import Image
from io import BytesIO
import base64
import sys
from main import *

# Set page configuration
st.set_page_config(layout="wide", page_title="Audiofy")


# Set page background
video_html = """
		<style>

		#myVideo {
		  position: fixed;
		  right: 0;
		  bottom: 0;
		  min-width: 100%; 
		  min-height: 100%;
		}

		.content {
		  position: fixed;
		  bottom: 0;
		  background: rgba(0, 0, 0, 0.5);
		  color: #f1f1f1;
		  width: 100%;
		  padding: 60px;
		}

		</style>	
		<video autoplay muted loop id="myVideo">
		  <source src="https://github.com/Bhavya1435/audiofytest/releases/download/ertyui/BackG.mp4")>
		  Your browser does not support HTML5 video.
		</video>
        """

st.markdown(video_html, unsafe_allow_html=True)

def load_image2audio():
    if "image2audio" not in st.session_state:
        st.session_state["image2audio"] = load_image2audio_model()
    return st.session_state["image2audio"]

def load_depth():
    if "depth" not in st.session_state:
        st.session_state["depth"] = load_depth_model()
    return st.session_state["depth"]

def load_segmentation():
    if "segmentation" not in st.session_state:
        st.session_state["segmentation"] = load_segmentation_model()
    return st.session_state["segmentation"]

#image2audio = load_image2audio()
#depth = load_depth()
#segmentation = load_segmentation()


# Function to convert image to byte stream
def convert_image_to_bytes(image):
    img_buffer = BytesIO()
    image.save(img_buffer, format="PNG")
    return img_buffer.getvalue()

# Function to get download link for WAV file
def get_wav_download_link(wav_file_path):
    with open(wav_file_path, "rb") as file:
        wav_file_data = file.read()
    b64 = base64.b64encode(wav_file_data).decode()
    href = f'<a href="data:audio/wav;base64,{b64}" download="audio.wav">Download generated audio</a>'
    return href

# Main function to run the app
def main():
    # Buttons to switch between sections
    
    section = st.sidebar.radio("", ('Upload Image', 'Download Audio'), index=0)

    if section == 'Upload Image':
        upload_image_section()
    elif section == 'Download Audio':
        download_audio_section()

# Section to upload image and remove background
def upload_image_section():
    st.write("## Audiofy")
    st.write(":sound: Try uploading an image to hear the sound it in the picture")
    st.write ("Step into the world of Audiofy, where images transform into vibrant soundscapes! Our innovative platform marries cutting-edge image processing with advanced deep learning techniques to craft immersive Dolby 5.1 surround sound clips like never before.\
                 Experience the Magic: Picture this : upload an image of your choice alongside a captivating text prompt, and watch as Audiofy works its magic. Our intricate blend of traditional image processing and state-of-the-art diffusion models seamlessly weaves together visuals and audio, creating a symphony for your senses.\
                Delve into the heart of Audiofy creative process! Witness the power of ViT models as they meticulously segment images, while encoder-only and transformer VAEs bring textual elements to life. Inspired by CoDi, our audio diffuser crafts spatial soundscapes that transport you to another dimension.")
    st.write("For example:")
    st.image('dog.jpg',width=300)

    # Sidebar
    uploaded_file = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image.save("try.jpg")
        st.image(image, width=450)
        prompt = st.text_input("", placeholder="Input text prompt here")
        if st.button("Generate Audio"):
            infer(segmentation, depth, image2audio, "try.jpg", "out.wav", prompt)
            st.audio("out.wav", "out.wav")


# Section to download audio
def download_audio_section():
    st.write("## Download WAV audio File")
    st.write("Click the button below to download your WAV audio file.")

    wav_file_path = "out.wav"
    # Download button
    st.markdown(get_wav_download_link(wav_file_path), unsafe_allow_html=True)


if __name__ == "__main__":
    main()