import streamlit as st
from rembg import remove
from PIL import Image
from io import BytesIO
import base64

# Set page configuration
st.set_page_config(layout="wide", page_title="Image Background Remover")

# Define the background image
background_image = "background.jpg"  # Provide the path to your background image

# Set page background
page_bg_img = f'''
<style>
body {{
background-image: url("{background_image}");
background-size: cover;
}}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

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
    st.sidebar.write("")
    st.sidebar.write("")
    section = st.sidebar.radio("", ('Upload Image', 'Download Audio'), index=0)

    if section == 'Upload Image':
        upload_image_section()
    elif section == 'Download Audio':
        download_audio_section()

# Section to upload image and remove background
def upload_image_section():
    st.write("## Remove background from your image")
    st.write(":dog: Try uploading an image to watch the background magically removed.")

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
    st.write("## Download WAV File")
    st.write("Click the button below to download your WAV file.")

    with open("StarWars60.wav", "rb") as f:
        data = f.read()
    st.download_button('Download story as mp3 file', data, 'StarWars60.wav')

if __name__ == "__main__":
    main()
