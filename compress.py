import gzip
import shutil

with open("ml_model_app/model.pkl", "rb") as f_in:
    with gzip.open("ml_model_app/model.pkl.gz", "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
