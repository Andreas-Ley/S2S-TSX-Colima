# Speckle2Speckle TerraSAR-X Filter

[![License: GPL v3](https://img.shields.io/badge/License-GPL_v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## Intro

This is a despeckling-filter for TerraSAR-X images based on convolutional networks. The code only contains the inference (application) part, not the training.

This is an initial test of applying the speckle2speckle deep learning approach to TerraSAR-X images. The filter has been trained on image of Colima only. We are expecting improvements with more data. 

In case of questions contact:

* Sébastien Valade (valade.sebastien@gmail.com)
* Andreas Ley (mail@andreas-ley.com)

If you use this in a publication, please consider citing us.

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new?hide_repo_select=true&ref=master&repo=570506234)

## Dependencies

The code is CPU only and thus does not need a GPU or associated drivers. It does, however, need AVX extensions which should be present in every x86(_64) cpu from the last decade. If in doubt, check with:

    cat /proc/cpuinfo | grep avx

The code is very much self contained and only needs an OpenMP capable C/C++ compiler and libTiff. A cmake file for building is also provided.

## Setup

Install the following dependencies:

* gcc
* cmake
* libtiff (development packages)

E.g. for ubuntu:

    sudo apt install gcc cmake libtiff5-dev

Or on fedora:

    sudo dnf install gcc cmake libtiff-devel

Create a subdirectory for an out-of-source build:

    mkdir buildRelease
    cd buildRelease

Run cmake:

    cmake .. -DCMAKE_BUILD_TYPE=Release

Build:

    make -j 8

This should build an executable "applicationCPU" in the "buildRelease" directory.

## Usage

The program ingests a (geo-)tiff image which must be the complex valued hh component in radar coordinates. Usually this is stored as two 16-bit integer values. The program can also ingest other data types (e.g. floats) but the value range/scaling must be the same.

The program outputs the filtered log-intensity values as a tiff image with float values which is intended for further processing. For visualization purposes it also outputs 8-bit images of the input and output. These are only intended for visualization and are not to be used for further processing. If you base your research on low quality 8-bit images with arbitrary tonemapping, you have noone to blame but yourself!

The program takes two arguments. The filename of the source tiff image and a prefix for the output. Running it like so:

    ./applicationCPU /path/to/Popo/track_136_ASC_spot_090_20190425_004720.tif Popo_track_136_ASC_spot_090_20190425_004720

will produce these three files:

* Popo_track_136_ASC_spot_090_20190425_004720.tif   -- Filtered log intensity output
* Popo_track_136_ASC_spot_090_20190425_004720_before.tif   -- Visualization of input image
* Popo_track_136_ASC_spot_090_20190425_004720_after.tif   -- Visualization of output image

## License

    Speckle2Speckle based despeckling filter for TerraSAR-X Spotlight mode
    Copyright (C) 2020  Andreas Ley <mail@andreas-ley.com>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
