import cv2 as cv
import numpy as np

# This script works best when used for 4k (3840x2160) images.

crt_mask_width = 3840   # must be multiple of 8
crt_mask_height = 2160  # must be multiple of 4
crt_number_of_scanlines = 200

def blit(target, img, x, y):
    assert target.shape[2] == img.shape[2]

    img_width = img.shape[1]
    img_height = img.shape[0]

    target[y:y+img_height, x:x+img_width] = img

def generate_shadowmask(width, height):
    shadowmask = cv.imread("subpixles_4k.png")
    shadowmask_h, shadowmask_w, shadowmask_channels = shadowmask.shape

    result_image = np.zeros((height, width, 3), np.uint8)

    count_x = width // shadowmask_w
    count_y = height // shadowmask_h

    for x in range(count_x):
        for y in range(count_y):
            blit(result_image, shadowmask, shadowmask_w * x, shadowmask_h * y)
        
    return result_image

def generate_scanlines(width, height, scanline_count):
    scanline_segment = cv.imread("scanline_smooth.png")
    result_image = np.zeros((width, height ,3), np.uint8)

    scanline_segment_h, scanline_segment_w, scanline_segment_c = scanline_segment.shape

    scanline_h = height / scanline_count
    scale_y = scanline_h / scanline_segment_h

    M = np.zeros((2,3), np.float)
    M[0,0] = 1       # scale x
    M[1,1] = scale_y # scale y
    M[0,2] = 0   # offset x
    M[1,2] = 0   # offset y

    img = cv.warpAffine(scanline_segment, M, (width, height), result_image, cv.INTER_LANCZOS4, cv.BORDER_WRAP )
    return img



shadowmask = generate_shadowmask(crt_mask_width, crt_mask_height)
scanlines = generate_scanlines(crt_mask_width, crt_mask_height, crt_number_of_scanlines)
scanlines = scanlines.astype("float") / 255.0
cv.imwrite("crt_template.png", shadowmask * scanlines)
