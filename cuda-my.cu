#include <cuda_runtime.h>
#include <iostream>
#include <vector>


// Your Kernel Function
__global__ void euclidian_distance(float2* p1, float2* p2, float* distance, int count) {
    // Calculate the index of the point my thread is working on
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    // Check if thread is in-range before reading data
    if (index < count) {
        // Compute the Euclidian distance between two points
        float2 dp = make_float2(p2[index].x - p1[index].x, p2[index].y - p1[index].y);
        float dist = sqrtf(dp.x * dp.x + dp.y * dp.y);
        // Write out the computed distance
        distance[index] = dist;
    }
}

// Error checking macro for simplicity
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

void fill_test_data(std::vector<float2>& v1, std::vector<float2>& v2) {
    int count = v1.size();
    for(int i = 0; i < count; i++) {
        v1[i] = make_float2(0.0f, 0.0f);
        v2[i] = make_float2(static_cast<float>(i), 0.0f); // Distance should be 'i'
    }
}

void test_on_cpu(std::vector<float> distances, const std::vector<float2>& v1, const std::vector<float2>& v2) {
    for(int i = 0; i < v1.size(); i++) {
        float dx = v1[i].x - v2[i].x;
        float dy = v1[i].y - v2[i].y;

        float dist = std::sqrt(dx*dx + dy*dy);
        distances[i] = dist;
    }
}

void test_on_cuda(std::vector<float>& distances, const std::vector<float2>& v1, const std::vector<float2>& v2) {
    // const int count = 100000; // Number of point pairs to process

    const int count =  distances.size();
    size_t size_float2 = count * sizeof(float2);
    size_t size_float = count * sizeof(float);

    // 1. Prepare data on the Host (CPU)

    // Initialize data (example: p1 at origin, p2 along x-axis)

    // 2. Allocate memory on the Device (GPU)
    float2 *d_p1, *d_p2;
    float *d_distance;

    CUDA_CHECK(cudaMalloc((void**)&d_p1, size_float2));
    CUDA_CHECK(cudaMalloc((void**)&d_p2, size_float2));
    CUDA_CHECK(cudaMalloc((void**)&d_distance, size_float));

    // 3. Copy data from Host to Device
    CUDA_CHECK(cudaMemcpy(d_p1, v1.data(), size_float2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_p2, v2.data(), size_float2, cudaMemcpyHostToDevice));

    // 4. Launch the Kernel
    // Define block size (threads per block)
    int threadsPerBlock = 256;
    // Calculate grid size (number of blocks) needed
    int blocksPerGrid = (count + threadsPerBlock - 1) / threadsPerBlock;

    // <<< Grid Size, Block Size >>> (configuration)
    euclidian_distance<<<blocksPerGrid, threadsPerBlock>>>(d_p1, d_p2, d_distance, count);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // 5. Wait for the GPU to finish and copy results back
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(distances.data(), d_distance, size_float, cudaMemcpyDeviceToHost));

    // 6. (Optional) Print first few results to verify
    // std::cout << "First 5 distances: ";
    // for(int i = 0; i < 5; i++) {
    //     std::cout << distances[i] << " ";
    // }
    // std::cout << std::endl;

    // 7. Free Device memory
    cudaFree(d_p1);
    cudaFree(d_p2);
    cudaFree(d_distance);
}



int main(int argc, char* argv[]) {
    std::string mode = argv[1];
    std::string string_count = argv[2];

    int count = std::stoi(string_count);
    // int count = 200000000;
    std::vector<float2> v1(count);
    std::vector<float2> v2(count);
    std::vector<float> distances(count);

    fill_test_data(v1, v2);
    if (mode == "cpu") test_on_cpu(distances, v1, v2); 
    else test_on_cuda(distances, v1, v2);

    
    return 0;
}