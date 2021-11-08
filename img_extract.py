import fitz
import numpy as np


def get_img(pdf_path) -> np.ndarray:
    """Extract the image from a 1-page scan."""
    pdf  = fitz.open(pdf_path)
    blocks = pdf[0].get_text("dict")
    imgblocks = [b for b in blocks if b["type"] == 1]

    imb = imgblocks[0]
    size = (imb['height'], imb['width'], -1)

    img = np.frombuffer(imb["image"], dtype=np.uint8).reshape(size)
    return img
