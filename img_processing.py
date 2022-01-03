from typing import Any, List, Tuple
import numpy as np
import cv2
from math import sin, tan
import matplotlib.pyplot as plt


def sgn(x: Any) -> int:
    if x == 0:
        return 0
    elif x > 0:
        return 1
    else:
        return -1


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

    return np.stack([e1_parametric, e4_parametric, e2_parametric, e3_parametric],
            axis=0)


def get_intersection_points(lines: np.ndarray) -> np.ndarray:
    """Get corner points of four lines."""
    params_ids = [(i-1, i) for i in range(len(lines))]
    param_intersection_points = [(lines[i], lines[j]) for i,j in params_ids]

    points = []

    for line1, line2 in param_intersection_points:
        r1, theta1 = line1
        r2, theta2 = line2

        ## Inclination terms for both lines
        a1 = -1/tan(theta1)
        a2 = -1/tan(theta2)

        ## Free terms for both lines
        b1 = r1/sin(theta1)
        b2 = r2/sin(theta2)

        ## Points coordinates
        x = -(b2-b1)/(a2-a1)
        y = a1*x+b1

        points.append((int(x), int(y)))

    return points


def draw_points(img: np.ndarray, points: List[Tuple[int, int]]) -> np.ndarray:
    """
    Draw intersection points onto the image.
    """
    for point in points:
        img = cv2.circle(img, point, 10, (255, 0, 0), -1)

    return img


## Redundant. Switching to sliding windows
def get_page_center(points: List[Tuple[int, int]]) -> Tuple[int, int]:
    """Get center of page knowing the corners."""
    return np.mean(points, axis=0, dtype=int)


## Redundant. Switching to sliding windows
def march_corner(img: np.ndarray,
                 center: Tuple[int, int],
                 corner: Tuple[int, int],
                 margin: int) -> Tuple[int, int]:
    """
    Get inside corner corresponding to outside corner.
    Expects grayscale image.
    """
    reached_corner = False
    reached_horizontal_edge = False
    reached_vertical_edge = False

    step_x = sgn(corner[0] - center[0])
    step_y = sgn(corner[1] - center[1])

    x, y = center

    while not reached_corner:

        ## March towards horizontal edge (left/right)
        if not reached_horizontal_edge:
            x += step_x
            margin_x = x+(step_x*margin)
            if margin_x == img.shape[0] or margin_x < 0:
                reached_horizontal_edge = True
            elif img[margin_x, y] == 0:
                reached_horizontal_edge = True

        ## March towards vertical edge (top/bottom)
        if not reached_vertical_edge:
            y += step_y
            margin_y = y+(step_y*margin)
            if margin_y == img.shape[1] or margin_y < 0:
                reached_vertical_edge = True
            elif img[x, margin_y] == 0:
                reached_vertical_edge = True

        reached_corner = reached_horizontal_edge and reached_vertical_edge

    return (x, y)


## FIXME Check which corners are which !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def get_affine_src_points(img: np.ndarray,
                        corners: List[List[int]],
                        n_points: int,
                        window_width: int) -> List[List[int]]:
    """
    Perform the sliding window algorithm to get `n_points` points
    on each side for affine transformation.

    Expects binary image!!
    """
    ## Source points will be appended here
    ## Destination point order must match this order:
    ## Right edge from top to bottom (`n_points` points), then
    ## left edge from top to bottom (`n_points` points)
    src_pts = []

    ## ======================= RIGHT EDGE =====================================================

    ## Right edge
    window_height = int((corners[0/123][0/1] - corners[0/123][0/1]) / n_points) ## FIXME Check order of edges

    ## Precalculate first window left. Later calculated based on previous window
    left = corners[0/123][0/1] - int(window_width / 2)

    ## Go downwards from top corner to bottom corner
    for i in range(n_points):
        ## Calculate window
        window_upper = corners[0/123][0/1] + window_width*i

        ## Get window as section of image
        window = img[window_upper:window_upper+window_height, left:left+window_width]

        ## Get average index of right edge in the window
        ## HACK Calculated using `np.count_nonzero()` since the white region in the window should
        ## be solid and starting at the left edge of the window
        ## Point coordinate is then calculated as the mean of the edge and the middle of the window
        edge_indices = np.count_nonzero(window, axis=1)
        point_y = left + np.mean(edge_indices, dtype=int)
        point_x = int(window_upper + window_height*(i+1)/2)
        src_pts.append((point_x, point_y))

        ## Reposition window horizontally to ensure it does not go out of bounds
        ## Left in the next window is moved to match this window's edge point
        left = point_y - int(window_width/2)


    ## ======================= LEFT EDGE ======================================================


    ## Left edge
    window_height = int((corners[0/123][0/1] - corners[0/123][0/1]) / n_points) ## FIXME Check order of edges

    ## Precalculate first window left. Later calculated based on previous window
    left = corners[0/123][0/1] - int(window_width / 2)

    ## Go downwards from top corner to bottom corner
    for i in range(n_points):
        ## Calculate window
        window_upper = corners[0/123][0/1] + window_width*i

        ## Get window as section of image
        window = img[window_upper:window_upper+window_height, left:left+window_width]

        ## Get average index of right edge in the window
        ## HACK Calculated using `np.count_nonzero()` since the white region in the window should
        ## be solid and starting at the left edge of the window
        ## Point coordinate is then calculated as the mean of the edge and the middle of the window
        edge_indices = np.count_nonzero(window, axis=1)
        point_y = left + window_width - np.mean(edge_indices, dtype=int) ## Important change here (must be calculated differently)
        point_x = int(window_upper + window_height*(i+1)/2)
        src_pts.append((point_x, point_y))

        ## Reposition window horizontally to ensure it does not go out of bounds
        ## Left in the next window is moved to match this window's edge point
        left = point_y - int(window_width/2)




"""
Processing path:
1. Low thresh (global)
2. Closing
3. Median blur (large `ksize`)
4. Get largest area contour
5. Draw contour
6. Hough lines
7. Group lines into four edges
8. Get average parametric repr of each edge
9. Get corners as edge intersections
10. Get center of page
11. Get inside corners
-------------------------------------------
TODO
12. Sliding windows
13. Warp matrix

"""


orig = cv2.imread("thresh_closed_blurred.jpg", cv2.IMREAD_GRAYSCALE)
img = cv2.imread("contours.jpg", cv2.IMREAD_GRAYSCALE)
lines = get_hough_lines(cv2.imread("CNT.jpg", cv2.IMREAD_GRAYSCALE))
lines = average_lines(lines)
points = get_intersection_points(lines)
center = get_page_center(points)
corner = march_corner(orig, center, points[3], 100)
img = draw_hough_lines(np.stack([img, img, img], axis=-1), lines)
img = draw_points(img, points)
img = draw_points(img, [corner[::-1]])
cv2.imwrite("corner.jpg", img)

