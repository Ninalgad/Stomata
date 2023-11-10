FROM python:3.7

RUN apt update
RUN apt install gcc g++ git

# Create a /work directory within the container, copy everything from the
# build directory and switch there.
RUN mkdir /work
COPY requirements.txt /work/requirements.txt
WORKDIR /work


RUN pip install -r requirements.txt

COPY .. /work

# test and train scripts should be executable within the container.
RUN chmod +x test.sh
RUN chmod +x train.sh

CMD sh
# run the container with "docker run -it <image_name>" then in the shell call train.sh and test.sh scripts
