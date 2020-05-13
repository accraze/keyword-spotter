FROM python:3.7-slim
WORKDIR /app
ADD . /app
RUN apt-get update && apt-get install -y libsndfile1 libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg 
RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["hypercorn"  , "-b", "0.0.0.0:8000", "app:app"]
