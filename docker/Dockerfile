FROM python:3.10-slim

WORKDIR /src

COPY requirements.txt .

USER root
RUN apt-get update && \
    apt-get install -y python3-pip swig cmake libopenmpi-dev python3-dev zlib1g-dev libgl1-mesa-glx wget build-essential libsdl2-dev

# Build and install TA-Lib from source (patch config.guess/config.sub for ARM64)
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    wget -O config.guess 'https://git.savannah.gnu.org/cgit/config.git/plain/config.guess' && \
    wget -O config.sub 'https://git.savannah.gnu.org/cgit/config.git/plain/config.sub' && \
    chmod +x config.guess config.sub && \
    ./configure --prefix=/usr && make && make install && \
    cd .. && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

RUN python3 -m pip install --break-system-packages -vvv -r requirements.txt -I
RUN python3 -m pip install --break-system-packages jupyterlab
RUN python3 -m pip install --break-system-packages pytest

# Add Tini. Tini operates as a process subreaper for jupyter. This prevents kernel crashes.
ENV TINI_VERSION=v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini

# copy the code instead of mounting
COPY . .

ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["jupyter", "lab", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--NotebookApp.token=''","--NotebookApp.password=''","--allow-root"]
