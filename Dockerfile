FROM harbor.sdp.kat.ac.za/dpp/katsdpcontim

USER root

# Update, upgrade and install packages
RUN apt-get update && \
    apt-get upgrade -y

ENV PACKAGES \
    xterm \
    git \
    libeigen3-dev \
    python3-tk \
    python3.8-dev \
    libboost-dev

RUN apt-get install -y $PACKAGES && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

USER kat

ENV PATH="$PATH_PYTHON3" VIRTUAL_ENV="$VIRTUAL_ENV_PYTHON3"
#RUN pip install --no-deps --upgrade pyregion 
RUN pip install aplpy --upgrade
RUN pip install interpolation --upgrade
RUN pip install astropy --upgrade
RUN pip install numba --upgrade


# Install katcbfsim and katsdpimager
RUN mkdir /home/kat/src
WORKDIR /home/kat/src
RUN git clone https://github.com/ska-sa/katcbfsim
RUN cd katcbfsim && pip install . --upgrade
RUN git clone https://github.com/ska-sa/katsdpmodels
RUN cd katsdpmodels && pip install . --upgrade
RUN git clone https://github.com/ska-sa/katsdpimager
RUN cd katsdpimager && git submodule update --init --recursive \
				    && pip install .[katdal] --upgrade

RUN mkdir /home/kat/sim

COPY --chown=kat:kat MK+ArrayConfig /home/kat/sim/MK+ArrayConfig

COPY --chown=kat:kat MK+PBeams /home/kat/sim/MK+PBeams

COPY --chown=kat:kat mk+sim /home/kat/sim/mk+sim

COPY --chown=kat:kat SKADS /home/kat/sim/SKADS

COPY --chown=kat:kat STATIC /home/kat/sim/STATIC

# Configure Obit/AIPS disks
RUN cfg_aips_disks.py

RUN chmod +x /home/kat/sim/mk+sim/obit_sim_mk+.py

WORKDIR /home/kat/sim/mk+sim
