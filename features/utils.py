from .features_extractor import FeatureExtractor
from pandarallel import pandarallel
from datetime import datetime
import cv2 as cv
import pandas as pd


def from_path(path: str, **kwargs) -> dict[str, float]:
    import os

    if os.name == "nt":
        from features.features_extractor import FeatureExtractor
        import cv2 as cv

    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    extractor = FeatureExtractor(img, **kwargs)

    return extractor.features


def from_dataframe(df, verbose=True, **kwargs):
    pandarallel.initialize(nb_workers=4, progress_bar=verbose, verbose=0)

    ts = datetime.now()
    dict_series = df["path"].parallel_apply(from_path, **kwargs)
    if verbose:
        print(f"\nELAPSED TIME: {datetime.now() - ts}\n\n")

    return pd.DataFrame.from_records(dict_series)
