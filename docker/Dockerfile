FROM tensorflow/tensorflow
RUN apt-get update
RUN apt-get install vim -y
RUN apt-get install git -y
RUN apt-get install libsm6 libxext6 libxrender1 -y
COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
