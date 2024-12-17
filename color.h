#ifndef COLORS_H
#define COLORS_H

#include <iostream>
#include <CL/sycl.hpp>
#include <vector>
#include <cmath>
#include <vector>
#include "img_process.h"

class ColorTransformations {
public:
    // Convert RGB to HSV
    static void rgbToHsv(Image& image, sycl::queue& queue) {
        sycl::buffer<Image::Pixel, 2> pixelBuffer(
            image.getPixels().data(), 
            sycl::range<2>(image.getHeight(), image.getWidth())
        );

        queue.submit([&](sycl::handler& cgh) {
            auto pixels = pixelBuffer.get_access<sycl::access::mode::read_write>(cgh);

            cgh.parallel_for<class RgbToHsvKernel>(
                sycl::range<2>(image.getHeight(), image.getWidth()), 
                [=](sycl::id<2> idx) {
                    // Convert RGB to HSV color space
                    float r = pixels[idx].r / 255.0f;
                    float g = pixels[idx].g / 255.0f;
                    float b = pixels[idx].b / 255.0f;

                    float maxVal = std::max({r, g, b});
                    float minVal = std::min({r, g, b});
                    float delta = maxVal - minVal;

                    // Compute Hue, Saturation, Value
                    float h = 0.0f, s = 0.0f, v = maxVal;

                    if (delta > 0.0f) {
                        s = (maxVal > 0.0f) ? (delta / maxVal) : 0.0f;

                        if (maxVal == r) {
                            h = (g - b) / delta;
                        } else if (maxVal == g) {
                            h = 2.0f + (b - r) / delta;
                        } else {
                            h = 4.0f + (r - g) / delta;
                        }

                        h *= 60.0f;
                        if (h < 0.0f) h += 360.0f;
                    }

                    // Store back in pixel (using R, G, B as H, S, V)
                    pixels[idx].r = static_cast<uint8_t>(h / 360.0f * 255.0f);
                    pixels[idx].g = static_cast<uint8_t>(s * 255.0f);
                    pixels[idx].b = static_cast<uint8_t>(v * 255.0f);
                }
            );
        });
    }

    // Color Histogram Equalization
    static void equalizeHistogram(Image& image, sycl::queue& queue) {
        // Compute histogram in parallel
        std::vector<int> histogram(256, 0);
        
        sycl::buffer<int, 1> histogramBuffer(histogram.data(), histogram.size());
        
        // Parallel histogram computation
        queue.submit([&](sycl::handler& cgh) {
            auto hist_acc = histogramBuffer.get_access<sycl::access::mode::write>(cgh);
            auto pixels = sycl::buffer<Image::Pixel, 2>(
                image.getPixels().data(), 
                sycl::range<2>(image.getHeight(), image.getWidth())
            ).get_access<sycl::access::mode::read>(cgh);

            cgh.parallel_for<class HistogramKernel>(
                sycl::range<2>(image.getHeight(), image.getWidth()),
                [=](sycl::id<2> idx) {
                    // Compute luminance and increment histogram
                    uint8_t luma = static_cast<uint8_t>(
                        0.299f * pixels[idx].r + 
                        0.587f * pixels[idx].g + 
                        0.114f * pixels[idx].b
                    );
                    sycl::atomic_ref<int, sycl::memory_order::relaxed, 
                        sycl::memory_scope::device> atomic_hist(hist_acc[luma]);
                    atomic_hist.fetch_add(1);
                }
            );
        });

        // Compute cumulative histogram and equalization
        // Apply equalized values back to image
    }
}

#endif