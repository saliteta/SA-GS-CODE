'''
    This software is developed by Butian Xiong
    All rights reserved to FNII (Future Network of Intelligence Institude)
    Contact butianxiong@link.cuhk.edu.cn if one has any question
'''

import numpy as np
import cv2
from matplotlib import pyplot as plt

'''
    We need to have the following useage
    - Complexity Measurement
    - Visualization 
    - Method Selection

    We provide three function here:
    - Entropy
    - Edge Complexity
    - FFT Analysis 

    Additional Information 
    - We provide data loader and mask
    - We provide batch loading and processing module
    - We use CUDA to calculate the mean and variance    
'''

class image_complexity():
    def __init__(self, image_path:str) -> None:
        self.image_path = image_path
        self.image = cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED)
            # Load the image

        # Check if image has an alpha channel
        if self.image.shape[2] == 4:
            # Convert from BGRA to BGR (discard alpha channel)
            self.rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGRA2BGR)
            self.valid_pixel_number = np.sum(self.image[:, :, 3] > 0)

        # Now convert to grayscale

        # Proceed with further processing
        # (e.g., compute histogram, apply edge detection, etc.)

        self.img_gray = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2GRAY)

    def entropy_complexity(self):
        hist = cv2.calcHist([self.img_gray],[0],None,[256],[0,256])
        hist_norm = hist.ravel()/hist.sum()

        # Calculate entropy
        entropy = -np.sum(hist_norm*np.log2(hist_norm + 1e-9))  # add a small constant to avoid log(0)

        return entropy/self.valid_pixel_number, hist, entropy
    
    def edge_complexity(self):
        edges = cv2.Canny(self.rgb_image, 30, 260)  # Canny edge detection
        edge_count = np.sum(edges > 0)

        edge_density = edge_count / self.valid_pixel_number
        return edge_density, edges, edge_count

    def fft_complexity(self):
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
        mag = fourier_transform(self.img_gray)
        f_mu = weighted_mean(magnitude_spectrum=mag)
        return f_mu/self.valid_pixel_number, mag, f_mu
    
    def visualization(self, vis_picture, method:str):
        '''
        - Color Histogram image for entropy
        - Canny edge for edge complexity
        - Spectrum for FFT complexity
        '''
            
        plt.subplot(121), plt.imshow(self.img_gray, cmap='gray')
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])

        if method == 'entropy':
            plt.subplot(122)
            plt.plot(vis_picture)
            plt.xlim([0, 256])
            plt.grid(True)
            plt.xlabel('Pixel Value')
            plt.ylabel('Frequency')
            plt.title(method), plt.xticks([]), plt.yticks([])

        elif method == 'canny':
            plt.subplot(122)
            plt.imshow(vis_picture, cmap='gray')
            plt.title('Canny Edges')
            plt.axis('off')
            plt.title(method), plt.xticks([]), plt.yticks([])

        elif method == 'fft':
            plt.subplot(122)
            plt.imshow(vis_picture, cmap='gray')
            plt.title('specturm Edges')
            plt.axis('off')
            plt.title(method), plt.xticks([]), plt.yticks([])

        plt.show()

