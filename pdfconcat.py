import fitz
import numpy as np

from datetime import datetime
from typing import List


def rescale_A4(page: np.ndarray, dpi: int) -> np.ndarray:
    """Rescale a page to fit an A4 format page."""
    pass
    ## TODO Rescale & check pixel values for A4


def concat(pages: List[np.ndarray]) -> fitz.Document:
    """
    Concatenate & rescale images and compile them to a pdf document.
    """
    pages_rescaled = []
    pdf = fitz.Document()

    for page in pages:
        resc = rescale_A4(page)
        h, w, _ = resc
        page = pdf.new_page(width=w, height=h)
        ## TODO Add image to page

    return pdf


def save_document(pdf: fitz.Document) -> None:
    """Save the pdf document, generating the filename by time."""
    timeformat = "%d_%m_%Y__%H_%M_%S"
    fname = 'SCAN_' + datetime.strftime(timeformat) + '.pdf'

    pdf.save(fname)



