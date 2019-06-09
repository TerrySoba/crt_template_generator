import cv2 as cv
import numpy as np


def blit(target, img, x, y):
    assert target.shape[2] == img.shape[2]

    img_width = img.shape[1]
    img_height = img.shape[0]

    target[y:y+img_height, x:x+img_width] = img

def generate_shadowmask(width, height):
    shadowmask = cv.imread("subpixles_4k.png")
    shadowmask_h, shadowmask_w, shadowmask_channels = shadowmask.shape

    # print("shadow: {}x{}".format(shadowmask_w, shadowmask_h))

    result_image = np.zeros((height,width,3), np.uint8)

    count_x = width // shadowmask_w
    count_y = height // shadowmask_h

    # print("counts {}:{}".format(count_x, count_y))

    for x in range(count_x):
        for y in range(count_y):
            blit(result_image, shadowmask, shadowmask_w * x, shadowmask_h * y)
        
    return result_image

# def generate_shadowmask(width, height, scanline_count):
#     scanline_segment = cv.imread("subpixels_big.png")
#     result_image = np.zeros((height, width, 3), np.uint8)

#     scanline_segment_h, scanline_segment_w, scanline_segment_c = scanline_segment.shape

#     scanline_h = height / scanline_count
#     scale_y = scanline_h / scanline_segment_h


#     M = np.zeros((2,3), np.float)
#     M[0,0] = scale_y # scale x
#     M[1,1] = scale_y # scale y
#     M[0,2] = 0   # offset x
#     M[1,2] = 0   # offset y

#     img = cv.warpAffine(scanline_segment, M, (width, height), result_image, cv.INTER_AREA, cv.BORDER_WRAP )
#     # cv.imwrite("output.png", img)
#     return result_image

def generate_scanlines(width, height, scanline_count):
    scanline_segment = cv.imread("scanline_smooth.png")
    result_image = np.zeros((1920 * 2,1080 * 2,3), np.uint8)

    scanline_segment_h, scanline_segment_w, scanline_segment_c = scanline_segment.shape

    scanline_h = height / scanline_count
    scale_y = scanline_h / scanline_segment_h


    M = np.zeros((2,3), np.float)
    M[0,0] = 1       # scale x
    M[1,1] = scale_y # scale y
    M[0,2] = 0   # offset x
    M[1,2] = 0   # offset y

    img = cv.warpAffine(scanline_segment, M, (width, height), result_image, cv.INTER_LANCZOS4, cv.BORDER_WRAP )
    # cv.imwrite("output.png", img)
    return img

shadowmask = generate_shadowmask(1920 * 2, 1080 * 2)
#cv.imshow("image", shadowmask)
#cv.waitKey(0)


scanlines = generate_scanlines(1920 * 2, 1080 * 2, 200)
scanlines = scanlines.astype("float") / 255.0
cv.imwrite("crt_template.png", shadowmask * scanlines)
# cv.imshow("image", scanlines)
#cv.waitKey(0)


