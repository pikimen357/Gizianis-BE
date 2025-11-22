import base64

def image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as img:
        encoded = base64.b64encode(img.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"
