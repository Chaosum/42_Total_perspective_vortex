from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pywt

class WaveletTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, wavelet='db4', level=3):
        self.wavelet = wavelet
        self.level = level

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        features = []
        for epoch in X:  # shape: (n_channels, n_times)
            epoch_features = []
            for channel in epoch:
                coeffs = pywt.wavedec(channel, self.wavelet, level=self.level)
                for c in coeffs:
                    epoch_features.append(np.mean(np.abs(c)))
                    epoch_features.append(np.std(c))
                    epoch_features.append(np.sum(np.square(c)))
            features.append(epoch_features)
        return np.array(features)