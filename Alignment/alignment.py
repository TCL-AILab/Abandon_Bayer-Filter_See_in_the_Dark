import cv2
import os
from PIL import Image
import numpy as np
import tqdm
import math


def alignImages(im1, im2):
    """
    im1: to be aligned (mono_image)
    im2: reference img (rgb_image)
    return: aligned im1 and homograph
    """
    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_MATCHES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im1.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return im1Reg, h

if __name__ == '__main__':
    MAX_MATCHES = 1000
    GOOD_MATCH_PERCENT = 0.15
    dirpath = './raw/'        # image file dictionary
    CLK = '48mp'
    my_files = [f.name for f in os.scandir(dirpath) if
                f.name.endswith('.raw') and f.name.startswith('M')]
    for one in tqdm.tqdm(range(400, 401)):
        # each scene use one exposure image pair to get the homograph
        # and align 8 exposure monochrome images with this homograph
        to_be_regiester = dirpath + 'M{:0>5d}_{}_0x8_0x1fff.jpg'.format(one, CLK)
        reference_img = dirpath + 'C{:0>5d}_{}_0x8_0x1fff.jpg'.format(one, CLK)

        rgb_im = np.array(Image.open(reference_img))
        gray_im = np.array(Image.open(to_be_regiester))
        imReg, h = alignImages(gray_im, rgb_im)  # to be reg, reference, raw to be reg

        height, width, channels = rgb_im.shape

        for expo in ['0x007f', '0x00ff', '0x01ff', '0x03ff', '0x07ff', '0x0fff', '0x1fff', '0x2fff']:
            try:
                # Code that may raise an error
                gray_raw = np.fromfile(dirpath + 'M{:0>5d}_{}_0x8_{}.raw'.format(one, CLK, expo), dtype=np.uint8)
                gray_raw = np.reshape(gray_raw, (1024, 1280))
                rawimReg = cv2.warpPerspective(gray_raw, h, (width, height))
                IMG = Image.fromarray(rawimReg)
                IMG.save(dirpath + 'M_Align_{:0>5d}_{}_0x8_{}.tif'.format(one, CLK, expo))

            except Exception as e:  # code to run if error occurs
                # code to run if error is raised
                print(dirpath + 'M{:0>5d}_{}_0x8_{}.raw'.format(one, CLK, expo), "ERROR", e)
                pass
            else:
                pass