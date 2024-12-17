#include <iostream>
#include <CL/sycl.hpp>
#include <omp.h>
#include <vector>
#include "img_process.h"

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