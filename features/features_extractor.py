from skimage.feature import local_binary_pattern
from radiomics.base import RadiomicsFeaturesBase
from importlib import import_module
import SimpleITK as sitk
import mahotas as mt
import numpy as np
import cv2 as cv
import radiomics
import logging


radiomics.setVerbosity(logging.ERROR)
_RadiomicsFeaturesBase = type[RadiomicsFeaturesBase]
_default_lbp = {"eps": 1e-7, "R": 3, "P": 8, "method": "uniform"}


def _get_feature_objects() -> dict[str, _RadiomicsFeaturesBase]:
    features = "FirstOrder", "GLCM", "GLRLM", "GLSZM", "NGTDM", "GLDM"
    objects = {}
    for feature in features:
        module = import_module(f"radiomics.{feature.lower()}")
        objects[feature] = getattr(module, f"Radiomics{feature}")
    return objects


class FeatureExtractor:
    features_objects = _get_feature_objects()

    def __init__(self, image: np.ndarray, **kwargs) -> None:
        if len(image.shape) == 3:
            self.img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        else:
            self.img = image

        self.mask = np.ones(self.img.shape)
        self.lbp_settings = kwargs.pop("lbp_settings", _default_lbp)
        self._features = {**self._radiomics(), **self._lbp(), **self._mahotas()}

    def _radiomics(self) -> dict[str, float]:
        img, mask = map(sitk.GetImageFromArray, [self.img, self.mask])
        data = {}
        for feature, obj in self.features_objects.items():
            extractor = obj(img, mask)
            features = extractor.execute()
            data.update({f"{feature}_{k}": v for [k, v] in features.items()})
        return data

    def _lbp(self) -> dict[str, float]:
        settings = self.lbp_settings.copy()
        eps = settings.pop("eps")

        lbp = local_binary_pattern(self.img, **settings)
        hist, _ = np.histogram(lbp)
        hist = hist.astype("float64")
        hist /= hist.sum() + eps

        return {f"lbp_{n}": pattern for [n, pattern] in enumerate(hist)}

    def _mahotas(self) -> dict[str, float]:
        features = {
            "haralick": mt.features.haralick(self.img, return_mean=True),
            "tas": mt.features.tas(self.img),
            "zernike": mt.features.zernike_moments(self.img, 10),
        }

        return {
            f"{feature}_{n}": value
            for [feature, values] in features.items()
            for [n, value] in enumerate(values)
        }

    @property
    def features(self) -> dict[str, float]:
        return self._features
