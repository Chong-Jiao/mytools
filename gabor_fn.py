import numpy as np
import matplotlib.pyplot as plt

# For visualizing the Gabor filter, part of codes is from https://en.wikipedia.org/wiki/Gabor_filter

def gabor_fn(sigma, theta, Lambda, psi, gamma):
    sigma_x = sigma
    sigma_y = float(sigma) / gamma

    # Bounding box
    nstds = 2 # Number of standard deviation sigma
    x1 = abs(nstds * sigma_x * np.cos(theta) - nstds * sigma_y * np.sin(theta))
    x2 = abs(nstds * sigma_x * np.cos(theta) + nstds * sigma_y * np.sin(theta))
    xmax = max(x1, x2)
    xmax = np.ceil(max(1, xmax))

    y1 = abs(nstds * sigma_x * np.sin(theta) + nstds * sigma_y * np.cos(theta))
    y2 = abs(-nstds * sigma_x * np.sin(theta) + nstds * sigma_y * np.cos(theta))
    ymax = max(y1, y2)
    ymax = np.ceil(max(1, ymax))
    xmin = -xmax
    ymin = -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

    # Rotation 
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi / Lambda * x_theta + psi)
    #xfilter = np.bitwise_and((np.abs(x_theta)<=sigma_x*nstds), (np.abs(y_theta)<=sigma_y*nstds))
    #gb = np.where(xfilter, 1, 0)
    return gb

sigma = 80
theta = 0*np.pi/6
Lambda = 50
psi = 0
gamma = 2
gb = gabor_fn(sigma, theta, Lambda, psi, gamma)

print(gb.shape)

plt.imshow(gb, cmap='gray')
plt.show()
