import streamlit as st
from rembg import remove
from PIL import Image
from io import BytesIO
import base64

st.set_page_config(layout="wide", page_title="Image Background Remover")

st.write("## Remove background from your image")
st.write(
    ":dog: Try uploading an image to watch the background magically removed. Full quality images can be downloaded from the sidebar. This code is open source and available [here](https://github.com/tyler-simons/BackgroundRemoval) on GitHub. Special thanks to the [rembg library](https://github.com/danielgatis/rembg) :grin:"
)
st.sidebar.write("## Upload and download :gear:")

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

# Download the fixed image
def convert_image(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im


def fix_image(upload):
    image = Image.open(upload)
    col1.write("Original Image :camera:")
    col1.image(image)

    fixed = remove(image)
    col2.write("Fixed Image :wrench:")
    col2.image(fixed)
    st.sidebar.markdown("\n")
    st.sidebar.download_button("Download fixed image", convert_image(fixed), "fixed.png", "image/png")


col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if my_upload is not None:
    if my_upload.size > MAX_FILE_SIZE:
        st.error("The uploaded file is too large. Please upload an image smaller than 5MB.")
    else:
        fix_image(upload=my_upload)
else:
    fix_image("./zebra.jpg")

def set_background():
    # Add HTML and CSS to set a video as background
    html_code = """
    <style>
    body {
        background-color: #f0f2f6; /* Set a background color */
    }

    .bg-video {
        position: fixed;
        top: 0;
        right: 0;
        bottom: 0;
        left: 0;
        overflow: hidden;
        z-index: -1;
        width: 100%;
        height: 100%;
    }

    #video-background {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        min-width: 100%;
        min-height: 100%;
        width: auto;
        height: auto;
        z-index: -100;
    }
    </style>
    <div class="bg-video">
        <video id="video-background" autoplay loop muted>
            <source src="your_video_path.mp4" type="video/mp4"> <!-- Replace "your_video_path.mp4" with your video file path -->
            Your browser does not support the video tag.
        </video>
    </div>
    """
    st.markdown(html_code, unsafe_allow_html=True)

def main():
    set_background()

    # Your Streamlit app content here
    st.title("Streamlit App with Custom Video Background")
    st.write("This is your Streamlit app content.")

if __name__ == "__main__":
    main()
