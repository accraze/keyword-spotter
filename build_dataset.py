import json
import os

import librosa

DATASET_PATH = "dataset"
JSON_PATH = "data.json"
SAMPLES_TO_CONSIDER = 22050  # 1 sec worth of audio


def build_dataset(dataset_path, json_path, n_mfcc=13,
                  hop_length=512, n_fft=2048):
    # data dict
    data = {
        'mappings': [],
        'labels': [],
        'MFCCs': [],
        'files': []
    }

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        if dirpath is not dataset_path:
            # update mappings
            category = dirpath.split('/')
            data['mappings'].append(category)
            print(f'Processing: {category}')

            # loop through all filenames and extract mfccs

            for f in filenames:
                # get file path
                file_path = os.path.join(dirpath, f)

                # load audio file
                signal, sr = librosa.load(file_path)

                # ensure audio file is 1 sec or more
                if len(signal) >= SAMPLES_TO_CONSIDER:

                    # make sure signal is at least 1 sec
                    signal = signal[:SAMPLES_TO_CONSIDER]

                    # extract mfccs
                    MFCCs = librosa.feature.mfcc(
                        signal, n_mfcc=n_mfcc, hop_length=hop_length,
                        n_fft=n_fft)

                    # store data
                    data['labels'].append(i-1)
                    data['MFCCs'].append(MFCCs.T.tolist())
                    data['files'].append(file_path)
                    print(f'{file_path}: {i-1}')

    # store in json file
    with open(json_path, 'w') as fp:
        json.dump(data, fp, indent=4)


if __name__ == '__main__':
    build_dataset(DATASET_PATH, JSON_PATH)
