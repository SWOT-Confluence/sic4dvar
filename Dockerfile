# Use a base image with the necessary runtime
FROM ubuntu:24.04

# Set environment variables to non-interactive
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    tzdata \
    build-essential \
    manpages-dev \
    wget \
    python3-pip \
    python3.12-venv \
    python3.12-dev \
    && rm -rf /var/lib/apt/lists/*


# Set the working directory
WORKDIR /app

SHELL ["/bin/bash", "-c"]

# Copy the compiled binary, .so files, and other necessary files to the container
RUN mkdir -p /app/src
COPY sic4dvar_algos /app/src/sic4dvar_algos
COPY sic4dvar_classes /app/src/sic4dvar_classes
COPY sic4dvar_functions /app/src/sic4dvar_functions
COPY lib /app/src/lib
COPY configs/specific_filters /app/src/configs/specific_filters

COPY sic4dvar_param_confluence.ini sic4dvar_param_confluence.ini
COPY sic4dvar_params.py /app/src/sic4dvar_params.py

COPY sic4dvar.py /app/src/sic4dvar.py
COPY requirements.txt /app/requirements.txt

RUN python3 -m venv /app/env
RUN /app/env/bin/pip install -r /app/requirements.txt

# Expose necessary volumes
VOLUME ["/app/input", "/app/output", "/app/logs"]

# Define the entry point for the container
# ENTRYPOINT ["/bin/bash"]
ENTRYPOINT ["/app/env/bin/python3", "/app/src/sic4dvar.py"]
