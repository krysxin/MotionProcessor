import json
import time
import io
from PIL import Image


def save_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def load_json(filename):
    with open(filename, 'r') as f:
        lines = json.load(f)

    return lines


def get_transparent_img_bytes(size=(50,50)):
    image = Image.new("RGBA", size, (0, 0, 0, 0))

    # Convert the image to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_bytes = img_byte_arr.getvalue()

    return img_bytes



class timer:
    def __init__(self, method_name="timed process"):
        self.method = method_name

    def __enter__(self):
        self.start = time.time()
        print(f"{self.method} starts")

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.time()
        print(f"{self.method} took {str(round(end - self.start, 2))}s")