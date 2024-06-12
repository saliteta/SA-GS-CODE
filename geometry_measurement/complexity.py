import cv2
import numpy as np
epsilon = 1e-9

class complexity():
    '''
        We will use GPU to accelerate the process
    '''
    def __init__(self) -> None:
        pass

    def edge_analysis(self, images):
        edge_densities = []
        valid_pixel = []
        for img in images:
            edges = cv2.Canny(img, 30, 260)  # Canny edge detection

            valid_pixel.append(np.count_nonzero(np.sum(img,axis=-1)))

            edge_count = np.sum(edges > 0)

            edge_density = edge_count  

            edge_densities.append(edge_density)
        
        return np.array(edge_densities), np.array(valid_pixel)

    def spectrum_analysis(self, images):
        def fourier_transform(image:np.ndarray):
            f = np.fft.fft2(image)
            fshift = np.fft.fftshift(f)

            # Step 3: Magnitude Spectrum (for visualization)
            magnitude_spectrum = 20 * np.log(np.abs(fshift) + epsilon)

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
        
        f_mus = []
        valid_pixel = []
        for img in images:
            valid_pixel.append(np.count_nonzero(np.sum(img,axis=-1)))
            mag = fourier_transform(img)
            f_mu = weighted_mean(magnitude_spectrum=mag)
            f_mus.append(f_mu)
        return np.array(f_mus)

    def entropy_analysis(self, images):
        entropy_values = []
        valid_pixel = []
        for img in images:
            valid_pixel.append(np.count_nonzero(np.sum(img,axis=-1)))

            hist = cv2.calcHist([img],[0],None,[256],[0,256])
            hist_norm = hist.ravel()/hist.sum()

            # Calculate entropy
            entropy = -np.sum(hist_norm*np.log2(hist_norm + 1e-9))  # add a small constant to avoid log(0)
            entropy_values.append(entropy)
        return np.array(entropy_values), np.array(valid_pixel)
    

