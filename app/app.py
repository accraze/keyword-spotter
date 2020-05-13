import os
import random

from quart import Quart, jsonify, request
from service import Keyword_Spotting_Service

app = Quart(__name__)


@app.route('/predict', methods=['POST'])
async def predict():
    # get audio file and save it
    files = await request.files
    audio_file = files['file']
    file_name = str(random.randint(0, 100000))
    audio_file.save(file_name)

    kss = Keyword_Spotting_Service()
    predict = kss.predict(file_name)

    # remove the audio file
    os.remove(file_name)

    # send back the predicted keyword
    data = {'keyword': predict}
    return jsonify(data)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)
