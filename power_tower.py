import numpy as np
from tqdm import tqdm
from PIL import Image
from utils import get_colour, is_close


# Vector of initial points in the complex plane
re_min, re_max = (-3.5, 5.5)
im_min, im_max = (-4.5, 4.5)
resolution = 200
initial_zs = (np.expand_dims(np.linspace(re_min, re_max, resolution), 0) + 1j * np.expand_dims(np.linspace(im_min, im_max, resolution), 1)).flatten()

# Iterate raising to the power
iterations = 200
record_history_length = 30
zs = initial_zs
trajectories = []
for i in tqdm(range(iterations)):
    zs = initial_zs**zs

    # Record the last 30 steps
    if iterations - i < record_history_length + 1:
        trajectories.append(zs)
trajectories = trajectories[::-1]

# Record periodicities
period = np.zeros_like(zs, dtype=np.int32)
for i in range(1, record_history_length, 1):
    close = is_close(zs, trajectories[i], atol=1e-12)
    # If we've found a repeat for a point we've not previously
    # found a repeat for, record the period
    period[np.logical_and(period == 0, close)] = i

# Image array, initially white
rgb = np.ones((resolution * resolution, 3)) * 255
# Diverging points are black
rgb[~np.isfinite(zs)] = np.array([0, 0, 0])
# Colour points for each periodicity
for i in range(1, record_history_length, 1):
    rgb[period == i] = get_colour(i - 1)

# Save the image
im = Image.fromarray(
    rgb.astype(np.uint8).reshape(resolution, resolution, 3)
)
im.save('power_tower.png')