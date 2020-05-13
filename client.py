import requests

URL = "http://127.0.0.1:80/predict"
TEST_AUDIO_FILE_PATH = 'test/left.wav'

if __name__ == '__main__':
    audio_file = open(TEST_AUDIO_FILE_PATH, 'rb')
    values = {'file': (TEST_AUDIO_FILE_PATH, audio_file, 'audio/wav')}
    res = requests.post(URL, files=values)
    data = res.json()
    print(f"Predicted keyword is: {data['keyword']}")
