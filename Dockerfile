FROM python:3.9

RUN apt-get update && apt-get install -y libglu1-mesa-dev freeglut3-dev mesa-common-dev cmake

ADD . /code
WORKDIR /code

RUN pip install -r requirements.txt
