import fitz

def get_img_name(pdf_path) -> str:
    """Extract the image from a 1-page scan."""
    pdf  = fitz.open(pdf_path)

    xref = pdf[0].get_images()[0]
    xref_num = xref[0]
    
    pix = fitz.Pixmap(pdf, xref_num)
    imname = pdf_path.split('.')[0] + '_0.jpg'

    pix.save(imname)
    return imname

