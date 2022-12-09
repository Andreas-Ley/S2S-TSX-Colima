/*
 * Speckle2Speckle based despeckling filter for TerraSAR-X Spotlight mode
 * Copyright (C) 2020  Andreas Ley <mail@andreas-ley.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "FloatImage.h"
#include "RunFilter.h"

#include <tiffio.h>
#include <cstdint>
#include <iostream>
#include <cmath>


#define TIFFTAG_GEOTIFF_METADATA (65000)



void getWidthHeight(TIFF *tif, unsigned &width, unsigned &height)
{
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);
}


template<typename BaseType, unsigned numChannels>
void readFile(TIFF *tif, const unsigned *channelList, FloatImage<numChannels, float> &dst)
{
    unsigned tileWidth = 0, tileHeight = 0;
    TIFFGetField(tif, TIFFTAG_TILEWIDTH, &tileWidth);
    TIFFGetField(tif, TIFFTAG_TILELENGTH, &tileHeight);

    unsigned width = 0;
    unsigned height = 0;

    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);
    
    dst.allocate(width, height);

    unsigned samplesPerPixel = 0;
    TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel);
    unsigned planarConfig = 0;
    TIFFGetField(tif, TIFFTAG_PLANARCONFIG, &planarConfig);

    
/*
    if ((samplesPerPixel != 1) && (planarConfig != PLANARCONFIG_CONTIG)) {
        std::cout << "samplesPerPixel " << samplesPerPixel << std::endl;
        std::cout << "planarConfig " << planarConfig << std::endl;
        throw std::runtime_error("Unsupported planar configuration!");
    }
*/
    for (unsigned i = 0; i < numChannels; i++)
        if (channelList[i] >= samplesPerPixel)
            throw std::runtime_error("More channels requested than are present in the file!");


    if (tileWidth != 0) {
        unsigned tileSize = TIFFTileSize(tif);
        if (planarConfig == PLANARCONFIG_CONTIG) {
            BaseType  *buf = (BaseType *)_TIFFmalloc(tileSize);

            for (unsigned y = 0; y < height; y += tileHeight)
                for (unsigned x = 0; x < width; x += tileWidth) {
                    if (TIFFReadTile(tif, buf, x, y, 0, 0) == -1)
                        throw std::runtime_error("Error reading tile!");

                    for (unsigned y_ = 0; y_ < tileHeight; y_++) {
                        if (y+y_ >= height) continue;
                        for (unsigned x_ = 0; x_ < tileWidth; x_++) {
                            if (x+x_ >= width) continue;

                            for (unsigned i = 0; i < numChannels; i++)
                                dst(x+x_, y+y_, i) = *(buf + (x_ + y_*tileWidth)*samplesPerPixel + channelList[i]);
                        }
                    }
                }

            _TIFFfree(buf);
        } else if (planarConfig == PLANARCONFIG_SEPARATE) {
            BaseType  *buf = (BaseType *)_TIFFmalloc(tileSize*numChannels);

            for (unsigned y = 0; y < height; y += tileHeight)
                for (unsigned x = 0; x < width; x += tileWidth) {
                    for (unsigned i = 0; i < numChannels; i++)
                        if (TIFFReadTile(tif, buf + tileSize*i/sizeof(BaseType), x, y, 0, channelList[i]) == -1)
                            throw std::runtime_error("Error reading tile!");

                    for (unsigned y_ = 0; y_ < tileHeight; y_++) {
                        if (y+y_ >= height) continue;
                        for (unsigned x_ = 0; x_ < tileWidth; x_++) {
                            if (x+x_ >= width) continue;

                            for (unsigned i = 0; i < numChannels; i++)
                                dst(x+x_, y+y_, i) = *(buf + tileSize*i/sizeof(BaseType)  + x_ + y_*tileWidth);
                        }
                    }
                }

            _TIFFfree(buf);
        } else throw std::runtime_error("Unknown planar configuration!");
    } else {
        const unsigned scanlineSize = TIFFScanlineSize(tif);

        if (planarConfig == PLANARCONFIG_CONTIG) {
            BaseType  *buf = (BaseType *) _TIFFmalloc(scanlineSize);

            for (unsigned y = 0; y < height; y++) {
                TIFFReadScanline(tif, buf, y);
                for (unsigned x = 0; x < width; x++) {
                    for (unsigned i = 0; i < numChannels; i++)
                        dst(x, y, i) = *(buf + x*samplesPerPixel + channelList[i]);
                }
            }

            _TIFFfree(buf);
        } else if (planarConfig == PLANARCONFIG_SEPARATE) {
            BaseType  *buf = (BaseType *) _TIFFmalloc(scanlineSize*numChannels);

            for (unsigned i = 0; i < numChannels; i++)
                for (unsigned y = 0; y < height; y++) {
                    TIFFReadScanline(tif, buf + scanlineSize*i/sizeof(BaseType), y, channelList[i]);
                    for (unsigned x = 0; x < width; x++) 
                        dst(x, y, i) = *(buf + scanlineSize*i/sizeof(BaseType) + x);
                }
            
            _TIFFfree(buf);
        } else throw std::runtime_error("Unknown planar configuration!");
    }
}



template<unsigned numChannels>
void readFile(TIFF *tif, const unsigned *channelList, FloatImage<numChannels, float> &dst)
{
    unsigned sampleFormat = 0;
    TIFFGetField(tif, TIFFTAG_SAMPLEFORMAT, &sampleFormat);
    unsigned bitsPerSample = 0;
    TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bitsPerSample);
    
    switch (sampleFormat) {
        case SAMPLEFORMAT_INT:
            switch (bitsPerSample) {
                case 32:
                    readFile<std::int32_t, numChannels>(tif, channelList, dst);
                break;
                case 16:
                    readFile<std::int16_t, numChannels>(tif, channelList, dst);
                break;
                case 8:
                    readFile<std::int8_t, numChannels>(tif, channelList, dst);
                break;
                default:
                    throw std::runtime_error("Bits per sample not handled!");
            }
        break;
        case SAMPLEFORMAT_UINT:
            switch (bitsPerSample) {
                case 32:
                    readFile<std::uint32_t, numChannels>(tif, channelList, dst);
                break;
                case 16:
                    readFile<std::uint16_t, numChannels>(tif, channelList, dst);
                break;
                case 8:
                    readFile<std::uint8_t, numChannels>(tif, channelList, dst);
                break;
                default:
                    throw std::runtime_error("Bits per sample not handled!");
            }
        break;
        case SAMPLEFORMAT_IEEEFP:
            readFile<float, numChannels>(tif, channelList, dst);
        break;
        default:
            throw std::runtime_error("Sample format not handled!");
    }
}





void renderPreview(const FloatImage1f &image, const char *filename)
{
    TIFF* tif = TIFFOpen(filename, "w");
    if (tif == nullptr)
        throw std::runtime_error(std::string("Error opening file for writing: ")+filename);
   
   
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 8);
    TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_UINT);
    TIFFSetField(tif, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
    TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_LZW);

    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, image.getWidth());  // set the width of the image
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH, image.getHeight());    // set the height of the image

    TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, 1);
    
    std::vector<unsigned char> lineBuffer;
    lineBuffer.resize(image.getWidth());
    
    
    for (unsigned y = 0; y < image.getHeight(); y++) {
        for (unsigned x = 0; x < image.getWidth(); x++) {
            unsigned r = std::min<int>(std::max<int>((image(x, y, 0)-6.0f)*40, 0), 255);
            
            lineBuffer[x] = r;
        }
        if (TIFFWriteScanline(tif, lineBuffer.data(), y, 0) < 0)
            throw std::runtime_error(std::string("Error while writing to file: ")+filename);
    }
    
    TIFFClose(tif);
}

void renderRawTif(const FloatImage1f &image, const char *filename, TIFF *metaDataSource)
{
    TIFF* tif = TIFFOpen(filename, "w");
    if (tif == nullptr)
        throw std::runtime_error(std::string("Error opening file for writing: ")+filename);
   
    
    if (metaDataSource != nullptr) {
        // for some reason this doesn't seem to be working...
        unsigned count;
        char *data;
        TIFFGetField(metaDataSource, TIFFTAG_GEOTIFF_METADATA, &count, &data);
        TIFFSetField(tif, TIFFTAG_GEOTIFF_METADATA, &count, &data);
    }
    
   
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 32);
    TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
    TIFFSetField(tif, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
    TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_LZW);

    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, image.getWidth());  // set the width of the image
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH, image.getHeight());    // set the height of the image

    TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, 1);
    
    std::vector<float> lineBuffer;
    lineBuffer.resize(image.getWidth());
    
    
    for (unsigned y = 0; y < image.getHeight(); y++) {
        for (unsigned x = 0; x < image.getWidth(); x++) {
            lineBuffer[x] = image(x, y, 0);
        }
        if (TIFFWriteScanline(tif, lineBuffer.data(), y, 0) < 0)
            throw std::runtime_error(std::string("Error while writing to file: ")+filename);
    }
    
    TIFFClose(tif);
}


int main(int argc, char **argv)
{
    if (argc < 3) {
        std::cout << "Usage: ./applicationCPU geoTiffInputFile outputFilePrefix" << std::endl;
        return 0;
    }
    
    try {
    
        std::string filename(argv[1]);
        std::string outputPrefix(argv[2]);
        
        FloatImage2f inputImage;
        unsigned channels[] = {0, 1};
        std::cout << "Opening image " << filename << std::endl;
        TIFF *tif = TIFFOpen(filename.c_str(), "r");
        if (tif == nullptr)
            throw std::runtime_error(std::string("Could not read tif file: ") + filename);
        try {
            std::cout << "Reading image" << std::endl;
            readFile<2>(tif, channels, inputImage);
            

            std::cout << "Computing log intensities" << std::endl;
            FloatImage1f inputLogIntensity;
            inputLogIntensity.allocate(inputImage.getWidth(), inputImage.getHeight());
            for (unsigned y = 0; y < inputImage.getHeight(); y++)
                for (unsigned x = 0; x < inputImage.getWidth(); x++) {
                    float real = inputImage(x, y, 0);
                    float imag = inputImage(x, y, 1);
                        
                    float itensity = real*real + imag*imag;
                    inputLogIntensity(x, y, 0) = std::log(1+itensity);
                }
                
                
            std::string inputPreviewFilename = outputPrefix + "_before.tif";
            std::cout << "Writing input preview image to " << inputPreviewFilename << std::endl;
            renderPreview(inputLogIntensity, inputPreviewFilename.c_str());
                
            FloatImage1f outputLogIntensity;

            std::cout << "Running filter" << std::endl;
            runFilter(inputLogIntensity, outputLogIntensity);
            
            std::string outputPreviewFilename = outputPrefix + "_after.tif";
            std::cout << "Writing output preview image to " << outputPreviewFilename << std::endl;
            renderPreview(outputLogIntensity, outputPreviewFilename.c_str());
            
            std::string outputFilename = outputPrefix + ".tif";
            std::cout << "Writing filtered float log intensity image to " << outputFilename << std::endl;
            renderRawTif(outputLogIntensity, outputFilename.c_str(), tif);

            TIFFClose(tif);
        } catch (...) {
            TIFFClose(tif);
            throw;
        }
    } catch (const std::exception &e) {
        std::cerr << "An error occured: " << e.what() << std::endl;
        return -1;
    }
    
    std::cout << "All done" << std::endl;
    
    return 0;

}



