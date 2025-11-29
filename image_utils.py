import requests
from io import BytesIO
from PIL import Image

SAMPLE_IMAGE_URL = "https://llava-vl.github.io/static/images/view.jpg"

def load_sample_image(url=SAMPLE_IMAGE_URL):
    """Load sample image from URL (RGB)"""
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert('RGB')
    return image

def load_sample_image_qnant(url=SAMPLE_IMAGE_URL):
    """Load sample image (raw) for quantization tests"""
    raw_image = Image.open(requests.get(url, stream=True).raw)
    return raw_image

# ===========================================================
#   MULTI-IMAGE LOADER
# ===========================================================
def load_images_from_urls(url_list):
    images = []
    for url in url_list:
        try:
            r = requests.get(url)
            img = Image.open(BytesIO(r.content)).convert("RGB")
            images.append(img)
            print(f"Loaded: {url}")
        except:
            print(f"Failed: {url}")
    return images
