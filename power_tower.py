import numpy as np
from tqdm import tqdm
from PIL import Image
import numexpr as ne
from utils import get_colour, is_close

np.random.seed(123)

# Vector of initial points in the complex plane
re_min, re_max = (-3.5, 5.5)
im_min, im_max = (-4.5, 4.5)

h_resolution = 1000
iterations = 500
sub_pixel_sample = 2
record_history_length = 500

h_resolution *= sub_pixel_sample
v_resolution = int(h_resolution * (im_max - im_min) / (re_max - re_min))
initial_zs = (np.expand_dims(np.linspace(re_min, re_max, h_resolution), 0) + 1j * np.expand_dims(np.linspace(im_min, im_max, v_resolution), 1)).flatten()

# Iterate raising to the power
zs = initial_zs
zs = ne.evaluate('initial_zs**zs')
for i in tqdm(range(iterations)):
    zs = ne.re_evaluate()

final_zs = np.copy(zs)

# Record periodicities
period = np.zeros_like(final_zs, dtype=np.int32)
for i in tqdm(range(record_history_length)):
    zs = ne.evaluate('initial_zs**zs')

    close = is_close(final_zs, zs, atol=1e-6)
    # If we've found a repeat for a point we've not previously
    # found a repeat for, record the period
    period[np.logical_and(period==0, close)] = i + 1

# Image array, initially white
rgb = np.ones((h_resolution * v_resolution, 3)) * 255
# Diverging points are black
rgb[~np.isfinite(zs)] = np.array([0, 0, 0])
# Colour points for each periodicity
for i in range(1, record_history_length, 1):
    rgb[period == i] = get_colour(i - 1)

# Save the image
im = Image.fromarray(
    rgb.astype(np.uint8).reshape(v_resolution, h_resolution, 3)
)
im = im.resize(
    (h_resolution // sub_pixel_sample, v_resolution // sub_pixel_sample),
    Image.BILINEAR
)
im.save('power_tower.png')