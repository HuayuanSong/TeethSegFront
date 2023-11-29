FROM python:3.9

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
RUN apt-get update && apt-get upgrade -y && apt-get install gcc -y

# Requirements for Pygco
RUN apt-get update && apt-get install libgl1 -y
RUN pip install --no-cache-dir -r /code/requirements.txt

RUN pip install numpy
RUN pip install cython==0.29.21

RUN apt-get update
RUN apt-get install --reinstall build-essential -y
RUN pip install pygco==0.0.16

# Make user
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user . $HOME/app

# Finish deployment and run Streamlit app
CMD ["streamlit", "run", "ⓘ_Introduction.py", "--server.port=7860", "--server.address=0.0.0.0"]
