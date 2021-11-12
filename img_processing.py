import numpy as np
import cv2
from math import ceil


def thresh(img: np.ndarray, thresh: int = 20) -> np.ndarray:
    """Thresh an image."""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[-1]




def cutoff_top_bottom(img: np.ndarray) -> np.ndarray:
    """Cut off the black bars on the bottom and top if they exist."""
    h = img.shape[0]
    topcnt = 0
    bottomcnt = 0
    for i in range(len(img)):
        do_break = True

        if not np.any(img[i]):
            topcnt += 1
            do_break = False
        
        if not np.any(img[h-i-1]):
            bottomcnt += 1
            do_break = False
    
    if topcnt + bottomcnt >= h:
        raise ValueError("Black or empty image passed to function!")
    return img[topcnt:h-bottomcnt]


def cutoff_left_right(img: np.ndarray) -> np.ndarray:
    """
    Cut off left and/or right black bars from the image.
    """




def close(img: np.ndarray) -> np.ndarray:
    """Remove ink spots from image."""
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, (3, 3), iterations=20)


def erode(img: np.ndarray) -> np.ndarray:
    """Dilate image."""
    return cv2.morphologyEx(img, cv2.MORPH_ERODE, (7, 7) , iterations=10)


def blur(img: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(img, (13, 13) ,0)


def get_tranform_points(img: np.ndarray) -> np.ndarray:
    """
    Get warp perspective tranform points 
    """


def medianblur(img: np.ndarray) -> np.ndarray:
    """Apply median blur to image"""
    return cv2.medianBlur(img, 15)


"""
Processing path:
1. Low thresh (global)
2. Cut off side bars
3. Closing
4. Media blur (large `ksize`)
5. Get largest area contour
# 6. [!] `cv2.minAreaRect`
6. Get corners of max inside rect
7. Move each corner outwards (keeping a padding distance between corner coords
and edge). E.g.:
- move corner to the right until it's 5px from right edge
- move corner up until it's 5px from top edge 
- while moving, adjust so that it is constantly 5px from right edge
- at the end, add 5px to top and right to fund corner
"""


img = cv2.imread("closed.jpg")
img = medianblur(img)
cv2.imwrite("median.jpg", img)



