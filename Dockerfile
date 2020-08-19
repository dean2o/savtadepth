FROM pytorch/pytorch

RUN apt-get update && apt-get install -y software-properties-common && apt-get update
RUN add-apt-repository -y ppa:git-core/ppa && apt-get update && apt-get install -y git libglib2.0-dev

COPY requirements.txt ./
RUN pip install -r requirements.txt

RUN pip install jupyterlab
