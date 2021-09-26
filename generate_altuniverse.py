import tqdm

from mapmaker.generate_image import get_althistory_image

for i in tqdm.trange(1, 21):
    get_althistory_image(i, prefix="alt-universe-examples")
