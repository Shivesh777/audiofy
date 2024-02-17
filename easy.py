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
page_bg_img = '''
<style>
body {
background-image: url("https://example.com/background.jpg");
background-size: cover;
}
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

# Main function to run the app
def main():
    # Buttons to switch between sections
    st.sidebar.write("")
    st.sidebar.write("")
    section = st.sidebar.radio("", ('Upload Image', 'Download Image'), index=0)

    if section == 'Upload Image':
        upload_image_section()
    elif section == 'Download Image':
        download_image_section()

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

            # Download button
            st.markdown(get_image_download_link(fixed_image), unsafe_allow_html=True)

# Section to download fixed image
def download_image_section():
    st.write("## Download Fixed Image")
    st.write(":arrow_down: Click the button below to download the fixed image.")

    # Download button
    st.markdown(get_image_download_link(fixed_image), unsafe_allow_html=True)

# Function to get download link for image
def get_image_download_link(image):
    buffered = convert_image_to_bytes(image)
    b64 = base64.b64encode(buffered).decode()
    href = f'<a href="data:file/png;base64,{b64}" download="fixed_image.png">Download fixed image</a>'
    return href

if __name__ == "__main__":
    main()
