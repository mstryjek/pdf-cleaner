import numpy as np
import cv2

from typing import Tuple


class ScanProcessor():
    """
    Class for processing scanned images and making them more presentable. Removes background and leaves
    """
    def __init__(self, cfg) -> None:
        self.CFG = cfg


    def thresh(self, img: np.ndarray) -> np.ndarray:
        """Threshold the image to single out page."""
        flag = cv2.THRESH_BINARY_INV if self.CFG.THRESH_INVERTED else cv2.THRESH_BINARY
        return cv2.threshold(img, self.CFG.THRESHOLD, 255, flag)[1] ## 0th element is thresh value


    def blur(self, img: np.ndarray) -> np.ndarray:
        """Blur image to reduce noise and improve threshing."""
        if self.CFG.BLUR_KERNEL == 0:
            return img
        
        kernel = tuple([self.CFG.BLUR_KERNEL]*2) if isinstance(self.CFG.BLUR_KERNEL, int) else tuple(self.CFG.BLUR_KERNEL)
        return cv2.blur(img, kernel)


    def get_rect_mask(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the bounding rect of the largest blob and the mask containing only the largest blob.
        Assumes `img` has been thresholded.
        Returns [rect, mask].
        """
        cnts, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        ## Largest contour
        cnt = max(cnts, key=cv2.contourArea)

        ## Bounding rect of largest contours
        rect = cv2.boundingRect(cnt)

        ## Create mask with only largest contour
        mask = np.zeros(img.shape, dtype=np.uint8)
        mask = cv2.drawContours(mask, [cnt], 0, [255, 255, 255], -1)

        return rect, mask


    def erode(self, img: np.ndarray) -> np.ndarray:
        """Erode an image as specified in CFG."""
        ksize = tuple([self.CFG.EROSION_KERNEL]*2) if isinstance(self.CFG.EROSION_KERNEL, int) else tuple(self.CFG.EROSION_KERNEL)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
        return cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel, self.CFG.EROSION_ITERATIONS)


    def crop_content(self, img: np.ndarray, mask: np.ndarray, rect: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crop content to fit as little background onto the final image as possible.
        """
        ## Unpack rect
        x, y, w, h = rect

        ## Crop image and mask
        ## If you're confused about the order of x,y refer to how OpenCV defines its axes
        img = img[y:y+h, x:x+w]
        mask = mask[y:y+h, x:x+w]

        return img, mask


    def draw_content_only(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Draw background as solid color."""
        ## Color is given as null, should be determined automatically (as mean of non-black pixels)
        if self.CFG.BACKGROUND_COLOR == None:
            img[mask == 0] = 0
            color = np.sum(img) / np.count_nonzero(img) ## Mean excluding 0-values
        ## Image is grayscale, color is BGR
        elif len(img.shape) == 2:
            color = np.mean(self.CFG.BACKGROUND_COLOR)
        ## Color is given as int
        elif isinstance(self.CFG.BACKGROUND_COLOR, int):
            color = [self.CFG.BACKGROUND_COLOR]*3
        else:
            color = self.CFG.BACKGROUND_COLOR

        img[mask == 0] = color

        return img


    def cut_off_margins(self, img: np.ndarray) -> np.ndarray:
        """Remove margins from images to cut off even more background."""
        margin = self.CFG.MARGIN
        if margin == 0:
            return img
        return img[margin:-margin+1, margin:-margin+1]


    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Comprehensive call to the processor. Gets raw scan image and performs full processing path,
        returns final image.
        """
        img = self.blur(img)

        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        threshed = self.thresh(gray)

        rect, mask = self.get_rect_mask(threshed)

        mask = self.erode(mask)

        img, mask = self.crop_content(img, mask, rect)

        img =  self.draw_content_only(img, mask)

        return self.cut_off_margins(img)

