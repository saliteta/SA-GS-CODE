import numpy as np
import cv2
from matplotlib import pyplot as plt
import argparse

def fourier_transform(image:np.ndarray):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    # Step 3: Magnitude Spectrum (for visualization)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    return magnitude_spectrum

def weighted_mean(magnitude_spectrum):
    # Iterate over the array and compute weights and sums

    center_x, center_y = magnitude_spectrum.shape[1] // 2, magnitude_spectrum.shape[0] // 2
    weighted_sum = 0
    total_weight = 0

    for y in range(center_y, magnitude_spectrum.shape[0]):
        for x in range(center_x, magnitude_spectrum.shape[1]):
            # Cartesian to polar
            radius = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            theta = np.arctan2(y - center_y, x - center_x) * 180 / np.pi

            # Directly compute weights and sums for the first quadrant
            weight = magnitude_spectrum[y, x] * radius
            weighted_sum += weight * magnitude_spectrum[y, x]
            total_weight += weight

    # Calculate weighted mean
    if total_weight == 0:
        return 0
    else:
        return weighted_sum / total_weight


def visualization(image, magnitude_spectrum):
    plt.subplot(121), plt.imshow(image, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

    plt.show()

def argparser():

    parser = argparse.ArgumentParser(description="Visualize and FFT to original image to specturm magnitude")
    parser.add_argument("--image_path", type=str, help="the path to image file")
    parser.add_argument("--visualize", type=str2bool, help="Whether to visualize it or not (true/false)", default=True)
    return parser

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    parser = argparser()

    args = parser.parse_args()

    print(f"image path: {args.image_path}")
    print(f"visualization: {args.visualize}")

    image = cv2.imread(args.image_path, 0)

    mag = fourier_transform(image)

    f_mu = weighted_mean(mag)

    print(f_mu)

    if args.visualize:
        visualization(image=image, magnitude_spectrum=mag)

if __name__ == "__main__":
    main()
