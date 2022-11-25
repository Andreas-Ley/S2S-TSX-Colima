/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2020  <copyright holder> <email>
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

#include "RunFilter.h"

#include "S2S_net_TsX_network.h"

#include <iostream>

void runFilter(const FloatImage1f &inputImage, FloatImage1f &output) {
    
    unsigned width = inputImage.getWidth();
    unsigned height = inputImage.getHeight();
    
    std::cout << "    " << width << " x " << height << std::endl;
    
    output.allocate(width, height);
       
    unsigned miniBatchSize = 16;

    unsigned inputCropSize = 512;
    unsigned outputCropSize = 512;

    unsigned overlap = 96;
    unsigned effectiveInputCropSize = inputCropSize - overlap;
    unsigned effectiveOutputCropSize = outputCropSize - overlap;
    
    unsigned cols = (width + effectiveOutputCropSize-1) / effectiveOutputCropSize;
    unsigned rows = (height + effectiveOutputCropSize-1) / effectiveOutputCropSize;
    
    int inputOffset = (inputCropSize - effectiveOutputCropSize)/2;

    std::cout << "    Crops: " << cols <<" x " << rows << std::endl;
    
    unsigned r = 0;
    unsigned c = 0;

    auto nextTile = [&]{
        if (++c == cols) {
            r++;
            c = 0;
        }
    };
    
    Tensor inputTensor = allocateTensor(inputCropSize, inputCropSize, 1, 1, miniBatchSize);
    Tensor outputTensor = allocateTensor(outputCropSize, inputCropSize, 1, 1, miniBatchSize);

    try {
        while (r < rows) {

            std::vector<std::pair<unsigned, unsigned>> crops;
            {
                for (unsigned i = 0; i < miniBatchSize; i++) {
                    if (r >= rows) break;
                    crops.push_back({c, r});
                    
                    std::cout << "    Fetching crop " << c << " " << r << std::endl;
                    for (unsigned y = 0; y < inputCropSize; y++)
                        for (unsigned x = 0; x < inputCropSize; x++) {
                            
                            unsigned x_ = std::min<int>(std::max<int>((int)(x + c * effectiveOutputCropSize) - inputOffset, 0), width-1);
                            unsigned y_ = std::min<int>(std::max<int>((int)(y + r * effectiveOutputCropSize) - inputOffset, 0), height-1);
                            

                            inputTensor.data[
                                    y * (inputTensor.W*inputTensor.C*inputTensor.N) +
                                    x * (inputTensor.C*inputTensor.N) +
                                    0 * (inputTensor.N) +
                                    i] = inputImage(x_, y_, 0);
                        }
                    
                    nextTile();
                }
            }
            
            S2S_net_TsX_network_forward(&inputTensor, &outputTensor, nullptr);
            
            {
                for (unsigned i = 0; i < crops.size(); i++) {
                    auto tile = crops[i];
                    for (unsigned y = overlap/2; y < outputCropSize-overlap/2; y++)
                        for (unsigned x = overlap/2; x < outputCropSize-overlap/2; x++) {
                            
                            unsigned x_ = x + tile.first * effectiveOutputCropSize - inputOffset;
                            unsigned y_ = y + tile.second * effectiveOutputCropSize - inputOffset;
                            if ((x_ < width) && (y_ < height)) {

                                output(x_, y_, 0) =  outputTensor.data[
                                                            y * (outputTensor.W*outputTensor.C*outputTensor.N) +
                                                            x * (outputTensor.C*outputTensor.N) +
                                                            0 * (outputTensor.N) +
                                                            i];
                            }
                        }
                }
            }
        }
        
        freeTensor(&inputTensor);
        freeTensor(&outputTensor);
    } catch (...) {
        freeTensor(&inputTensor);
        freeTensor(&outputTensor);
        throw;
    }
}
