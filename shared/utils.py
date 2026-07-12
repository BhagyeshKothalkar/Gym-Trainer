import base64
from io import BytesIO

import cv2


def image_to_base64(img_array) -> str:
    _, buffer = cv2.imencode(".jpg", img_array)
    return base64.b64encode(buffer).decode("utf-8")


def pil_image_to_base64(image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def save_image_bytes(img_bytes: bytes, save_name: str = "output.png") -> None:
    with open(save_name, "wb") as f:
        f.write(img_bytes)
    print(f"Saved image to {save_name}")
