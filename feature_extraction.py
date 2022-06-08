from skimage.feature import local_binary_pattern
from importlib import import_module
from typing import Union
import logging
import radiomics
import SimpleITK as sitk
import pandas as pd
import mahotas as mt
import numpy as np
import cv2 as cv


def _get_feature_objects():
    objects = []
    for feature in FEATURES:
        module = import_module(f"radiomics.{feature.lower()}")
        obj = getattr(module, f"Radiomics{feature}")
        objects.append(obj)
    return objects


_logger = logging.getLogger(f"radiomics")
_logger.setLevel(logging.ERROR)

FEATURES = ["FirstOrder", "GLCM", "GLRLM", "GLSZM", "NGTDM", "GLDM"]
_DEFAULT_LBP_SETTINGS = {"eps": 1e-7, "R": 3, "P": 8, "method": "uniform"}
_FEATURES_OBJECTS = _get_feature_objects()


class FeatureExtractor:
    def __init__(self, image: Union[np.ndarray, str], **kwargs) -> None:
        if len(image.shape) == 3:
            self.img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        else:
            self.img = image

        self.mask = np.ones(self.img.shape)
        self.lbp_settings = kwargs.pop("lbp", _DEFAULT_LBP_SETTINGS)
        self.features = ...

    def _get_haralick(self):
        ht_mean = mt.features.haralick(self.img, return_mean=True)
        data = pd.DataFrame(ht_mean).T
        return data.rename(columns=lambda x: f"haralick_{x}")

    def _get_radiomics(self):
        img, mask = map(sitk.GetImageFromArray, (self.img, self.mask))
        data = pd.DataFrame()
        for obj in _FEATURES_OBJECTS:
            extracted = obj(img, mask).execute()
            features = pd.DataFrame(extracted, index=[0], dtype="float64")
            data = data.merge(
                features,
                how="outer",
                left_index=True,
                right_index=True,
            )
        return data.rename(columns=lambda x: f"radiomics_{x}")

    def _get_lbp(self):
        settings = self.lbp_settings.copy()
        eps = settings.pop("eps")
        lbp = local_binary_pattern(self.img, **settings)
        hist, _ = np.histogram(lbp)
        hist = hist.astype("float64")
        hist /= hist.sum() + eps

        data = pd.DataFrame(hist, dtype="float64").T
        return data.rename(columns=lambda x: f"lbp_{x}")

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, _):
        if getattr(self, "_features", False):
            raise AttributeError("attribute can't be set manually")
        haralick = self._get_haralick()
        lbp = self._get_lbp()
        radiomics = self._get_radiomics()
        self._features = pd.concat((haralick, lbp, radiomics), axis=1)

    @staticmethod
    def from_path(path):
        image = cv.imread(path, cv.IMREAD_GRAYSCALE)
        extractor = FeatureExtractor(image)
        return extractor.features
