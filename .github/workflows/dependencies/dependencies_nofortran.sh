#!/usr/bin/env bash
#
# Copyright 2020 Axel Huebl
#
# License: BSD-3-Clause-LBNL

# search recursive inside a folder if a file contains tabs
#
# @result 0 if no files are found, else 1
#

set -eu -o pipefail

sudo apt-get update

sudo apt-get install -y --no-install-recommends\
    build-essential \
    g++             \
    libopenmpi-dev  \
    openmpi-bin     \
    python3         \
    python3-pip

python3 -m pip install -U pip
python3 -m pip install -U build packaging setuptools wheel
