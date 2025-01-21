import gzip
import shutil

with open("production/califonia_housing.pkl", "rb") as f_in:
    with gzip.open("production/califonia_housing.pkl.gz", "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
