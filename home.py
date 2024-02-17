import streamlit as st
from rembg import remove
from PIL import Image
from io import BytesIO
import base64

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

# Function to remove background from image
def remove_background(image):
    return remove(image)

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
    href = f'<a href="data:audio/wav;base64,{b64}" download="audio.wav">Download WAV file</a>'
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
    st.write(":dog: Try uploading an image to hear the sound it in the picture")

    # Sidebar
    uploaded_file = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_column_width=True)

        if st.button("Remove Background"):
            with st.spinner('Removing background...'):
                fixed_image = remove_background(image)
            st.image(fixed_image, caption="Background Removed", use_column_width=True)

# Section to download audio
def download_audio_section():
    st.write("## Download WAV audio File")
    st.write("Click the button below to download your WAV audio file.")

    wav_file_path = "dog.wav"
    # Download button
    st.markdown(get_wav_download_link(wav_file_path), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
