import librosa
import numpy as np
import tensorflow.keras as keras

MODEL_PATH = 'model.h5'
NUM_SAMPLES_TO_CONSIDER = 22050  # 1 sec


class _Keyword_Spotting_Service:

    model = None
    _mappings = [
        "right",
        "go",
        "no",
        "left",
        "stop",
        "up",
        "down",
        "yes",
        "on",
        "off"
    ]
    _instance = None

    def predict(self, file_path):
        # extract mfccs
        MFCCs = self.preprocess(file_path)  # (# segments, # coefficients)

        # convert 2d mfccs array into 4d array (# samples, # segments, # coefficients, # channels)
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

        # make prediction
        predictions = self.model.predict(MFCCs)
        predicted_index = np.argmax(predictions)
        predicted_keyword = self._mappings[predicted_index]

        return predicted_keyword

    def preprocess(self, file_path, n_mfcc=13, n_fft=2048, hop_length=512):
        # load audio file
        signal, sr = librosa.load(file_path)

        # ensure consistency in audio file
        if len(signal) > NUM_SAMPLES_TO_CONSIDER:
            signal = signal[:NUM_SAMPLES_TO_CONSIDER]

        # extract MFCCs
        MFCCs = librosa.feature.mfcc(
            signal, sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

        return MFCCs.T


def Keyword_Spotting_Service():
    # ensure that we only have 1 instance
    if _Keyword_Spotting_Service._instance is None:
        _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
        _Keyword_Spotting_Service.model = keras.models.load_model(MODEL_PATH)
    return _Keyword_Spotting_Service._instance


if __name__ == '__main__':
    kss = Keyword_Spotting_Service()

    keyword1 = kss.predict('test/down.wav')
    keyword2 = kss.predict('test/left.wav')

    print(f'Predicted keywords: {keyword1} , {keyword2}')
