import tqdm

from mapmaker.generate_image import get_gondor_image

for i in tqdm.tqdm([None, *range(1, 21)]):
    get_gondor_image(i, prefix="images/gondor-examples")
