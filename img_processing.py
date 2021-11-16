import numpy as np
import cv2
from math import ceil
import matplotlib.pyplot as plt



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


def contour(img: np.ndarray) -> np.ndarray:
    """Get the largest contour in the image."""
    contours = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]
    return max(contours, key=cv2.contourArea)


def get_points(contour: np.ndarray) -> np.ndarray:
    """Get bounding rect of largest contour."""
    x1, y1, x2, y2 = cv2.boundingRect(contour)
    return ((x1, y1), (x2, y2))


def get_page(img: np.ndarray) -> np.ndarray:
    """
    Perform full operation pipeline.
    """
    gray = thresh(img)
    closed = close(gray)
    blurred = medianblur(closed)
    cnt = contour(blurred)
    (xmin, ymin), (xmax, ymax) = get_points(cnt)
    return img[ymin:ymax, xmin:xmax]


def draw_hough_lines(img: np.ndarray, lines: np.ndarray) -> np.ndarray:
    """Draw detected hough lines on an image."""
    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 10000*(-b))
        y1 = int(y0 + 10000*(a))
        x2 = int(x0 - 10000*(-b))
        y2 = int(y0 - 10000*(a))

        img = cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

    return img


def get_hough_lines(img: np.ndarray) -> np.ndarray:
    """
    Get cleaned parametric version of Hough lines.
    Retshape is (n,2).
    """
    lines = cv2.HoughLines(img, 1, np.pi/180, 300)
    return np.squeeze(lines)


def average_lines(lines: np.ndarray) -> np.ndarray:
    """
    Split Hough lines into four edges and average each edge out.
    """
    rho_stdev = np.std(lines[:, 0])

    def edge1(lines: np.ndarray) -> np.ndarray:
        """
        Edge 1:
        - small values of rho (around 0)
        - theta around 0 (<pi/4) or pi (>3pi/4)
        """
        small_rho = lines[np.abs(lines[:, 0]) < rho_stdev/2]
        theta_topi4 = small_rho[small_rho[:, 1] < np.pi/4]
        theta_from3pi4 = small_rho[small_rho[:, 1] > 3*np.pi/4] - np.pi
        return np.concatenate([theta_from3pi4, theta_topi4], axis=0)

    def edge2(lines: np.ndarray) -> np.ndarray:
        """
        Edge 2:
        - large values of rho
        - theta around 0 (<pi/4) or pi (>3pi/4)
        """
        large_rho = lines[np.abs(lines[:, 0]) > rho_stdev/2]
        large_rho[:, 0] = np.abs(large_rho[:, 0])
        theta_topi4 = large_rho[large_rho[:, 1] < np.pi/4]
        theta_from3pi4 = large_rho[large_rho[:, 1] > 3*np.pi/4] - np.pi
        return np.concatenate([theta_from3pi4, theta_topi4], axis=0)

    def edge3(lines: np.ndarray) -> np.ndarray:
        """
        Edge 3:
        - small values of rho (around 0)
        - theta around pi/2 (>pi/4 and <3pi/4)
        """
        large_rho = lines[np.abs(lines[:, 0]) < rho_stdev/2]
        theta_topi4 = large_rho[large_rho[:, 1] > np.pi/4]
        theta_from3pi4 = theta_topi4[theta_topi4[:, 1] < 3*np.pi/4]
        return theta_from3pi4

    def edge4(lines: np.ndarray) -> np.ndarray:
        """
        Edge 4:
        - large values of rho
        - theta around pi/2 (>pi/4 and <3pi/4)
        """
        large_rho = lines[np.abs(lines[:, 0]) > rho_stdev/2]
        large_rho[:, 0] = np.abs(large_rho[:, 0])
        theta_topi4 = large_rho[large_rho[:, 1] > np.pi/4]
        theta_from3pi4 = theta_topi4[theta_topi4[:, 1] < 3*np.pi/4]
        return theta_from3pi4

    e1_parametric = np.mean(edge1(lines), axis=0)
    e2_parametric = np.mean(edge2(lines), axis=0)
    e3_parametric = np.mean(edge3(lines), axis=0)
    e4_parametric = np.mean(edge4(lines), axis=0)

    return np.stack([e1_parametric, e2_parametric, e3_parametric, e4_parametric],
            axis=0)


def get_intersection_points(lines: np.ndarray) -> np.ndarray:
    """Get corner points of four lines."""
    ...


"""
Processing path:
1. Low thresh (global)
3. Closing
4. Median blur (large `ksize`)
5. Get largest area contour
6. Draw contour
7. Hough lines
8. Group lines into four edges
9. Get average parametric repr of each edge
10. Get corners as edge intersections
"""

"""
TODO
- Get corner intersection points.
- Create marching algorithm, fixing the curvature of the scan.
"""


img = cv2.imread("CNT.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imwrite("lines.jpg", img)