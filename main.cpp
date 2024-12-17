#include <CL/sycl.hpp>
#include <omp.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>

class ImageProcessor {
public:
    struct Pixel {
        uint8_t r, g, b;
    };

    class Image {
    private:
        std::vector<Pixel> pixels;
        size_t width, height;

    public:
        Image(size_t w, size_t h) : width(w), height(h), pixels(w * h) {}

        void loadFromBuffer(const uint8_t* rawData) {
            #pragma omp parallel for
            for (size_t i = 0; i < width*height*3; i+=3) {
                pixels[i/3] = {
                    rawData[i],     // R
                    rawData[i+1], // G
                    rawData[i+2]  // B
                };
            }
        }

        // Save image to raw pixel buffer
        void saveToBuffer(uint8_t* rawData) const {
            #pragma omp parallel for
            for (size_t i = 0; i < width * height; i++) {
                rawData[i*3]     = pixels[i].r;
                rawData[i*3+1] = pixels[i].g;
                rawData[i*3+2] = pixels[i].b;
            }
        }

        size_t getWidth() const { return width; }
        size_t getHeight() const { return height; }
        std::vector<Pixel>& getPixels() { return pixels; }
        const std::vector<Pixel>& getPixels() const { return pixels; }
    };

    static void convertToGrayscale(Image& image, sycl::queue& queue) {
        size_t width = image.getWidth();
        size_t height = image.getHeight();
        
        sycl::buffer<Image::Pixel, 2> pixelBuffer(
            image.getPixels().data(), 
            sycl::range<2>(height, width)
        );

        queue.submit([&](sycl::handler& cgh) {
            auto pixels = pixelBuffer.get_access<sycl::access::mode::read_write>(cgh);

            cgh.parallel_for<class GrayscaleKernel>(
                sycl::range<2>(height, width), 
                [=](sycl::id<2> idx) {
                    auto& pixel = pixels[idx];
                    
                    uint8_t gray = static_cast<uint8_t>(
                        0.299f * pixel.r + 
                        0.587f * pixel.g + 
                        0.114f * pixel.b
                    );

                    pixel.r = pixel.g = pixel.b = gray;
                }
            );
        });

        // Wait for the kernel to complete
        queue.wait_and_throw();
    }

    static void adjustBrightness(Image& image, float factor, sycl::queue& queue) {
        size_t width = image.getWidth();
        size_t height = image.getHeight();
        
        sycl::buffer<Image::Pixel, 2> pixelBuffer(
            image.getPixels().data(), 
            sycl::range<2>(height, width)
        );

        queue.submit([&](sycl::handler& cgh) {
            auto pixels = pixelBuffer.get_access<sycl::access::mode::read_write>(cgh);

            cgh.parallel_for<class BrightnessKernel>(
                sycl::range<2>(height, width), 
                [=](sycl::id<2> idx) {
                    auto& pixel = pixels[idx];
                    
                    // Adjust brightness with clamping
                    pixel.r = std::min(255.0f, std::max(0.0f, pixel.r * factor));
                    pixel.g = std::min(255.0f, std::max(0.0f, pixel.g * factor));
                    pixel.b = std::min(255.0f, std::max(0.0f, pixel.b * factor));
                }
            );
        });

        queue.wait_and_throw();
    }

    // Parallel Edge Detection (Sobel Operator)
    static void detectEdges(Image& image, sycl::queue& queue) {
        size_t width = image.getWidth();
        size_t height = image.getHeight();
        
        // Create a new image to store edge-detected result
        Image edgeImage(width, height);
        
        sycl::buffer<Image::Pixel, 2> inputBuffer(
            image.getPixels().data(), 
            sycl::range<2>(height, width)
        );
        
        sycl::buffer<Image::Pixel, 2> outputBuffer(
            edgeImage.getPixels().data(), 
            sycl::range<2>(height, width)
        );

        queue.submit([&](sycl::handler& cgh) {
            auto input = inputBuffer.get_access<sycl::access::mode::read>(cgh);
            auto output = outputBuffer.get_access<sycl::access::mode::write>(cgh);

            cgh.parallel_for<class EdgeDetectionKernel>(
                sycl::range<2>(height, width), 
                [=](sycl::id<2> idx) {
                    // Skip border pixels
                    if (idx[0] == 0 || idx[0] == height-1 || 
                        idx[1] == 0 || idx[1] == width-1) {
                        output[idx].r = output[idx].g = output[idx].b = 0;
                        return;
                    }

                    // Sobel kernels for x and y directions
                    int gx = 0, gy = 0;
                    
                    // Compute Sobel gradients for each color channel
                    for (int c = 0; c < 3; ++c) {
                        // Sobel X kernel
                        gx = 
                            -1 * input[{idx[0]-1, idx[1]-1}+c] + 
                             1 * input[{idx[0]-1, idx[1]+1}+c] +
                            -2 * input[{idx[0], idx[1]-1}+c] + 
                             2 * input[{idx[0], idx[1]+1}+c] +
                            -1 * input[{idx[0]+1, idx[1]-1}+c] + 
                             1 * input[{idx[0]+1, idx[1]+1}+c];

                        // Sobel Y kernel
                        gy = 
                            -1 * input[{idx[0]-1, idx[1]-1}+c] + 
                            -2 * input[{idx[0]-1, idx[1]}+c] +
                            -1 * input[{idx[0]-1, idx[1]+1}+c] +
                             1 * input[{idx[0]+1, idx[1]-1}+c] + 
                             2 * input[{idx[0]+1, idx[1]}+c] +
                             1 * input[{idx[0]+1, idx[1]+1}+c];

                        // Compute magnitude
                        int magnitude = static_cast<int>(std::sqrt(gx * gx + gy * gy));
                        
                        // Clamp and assign to output
                        output[idx][c] = static_cast<uint8_t>(std::min(magnitude, 255));
                    }
                }
            );
        });

        queue.wait_and_throw();

        // Copy edge-detected image back to original image
        image = std::move(edgeImage);
    }

    // Example pipeline that combines multiple operations
    static void processImagePipeline(Image& image, sycl::queue& queue) {
        // Convert to grayscale
        convertToGrayscale(image, queue);

        // Adjust brightness
        adjustBrightness(image, 1.5f, queue);

        // Detect edges
        detectEdges(image, queue);
    }
};

// Main function to demonstrate usage
int main() {
    // Create a SYCL queue (default selector)
    sycl::queue queue;

    // Example image dimensions
    const size_t WIDTH = 800;
    const size_t HEIGHT = 600;

    // Allocate raw image buffer
    std::vector<uint8_t> rawImageData(WIDTH * HEIGHT * 3);
    
    // TODO: Load actual image data here
    // For demonstration, fill with some dummy data
    #pragma omp parallel for
    for (size_t i = 0; i < rawImageData.size(); ++i) {
        rawImageData[i] = static_cast<uint8_t>(i % 256);
    }

    // Create image from raw data
    ImageProcessor::Image image(WIDTH, HEIGHT);
    image.loadFromBuffer(rawImageData.data());

    try {
        // Process the image using the parallel pipeline
        ImageProcessor::processImagePipeline(image, queue);

        // Save processed image
        std::vector<uint8_t> processedImageData(WIDTH * HEIGHT * 3);
        image.saveToBuffer(processedImageData.data());

        std::cout << "Image processing completed successfully!" << std::endl;
    }
    catch (const sycl::exception& e) {
        std::cerr << "SYCL Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}