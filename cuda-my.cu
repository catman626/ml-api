#include<cuda_runtime.h>

__global__ void euclidian_distance(float2* p1, float2* p2, float* distance, int count) {
    // Calculate the index of the point my thread is working on
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    // Check if thread is in-range before reading data
    if (index < count) {
        // Compute the Euclidian distance between two points
        float2 dp = p2[index] - p1[index];
        float dist = sqrtf(dp.x * dp.x + dp.y * dp.y);
        // Write out the computed distance
        distance[index] = dist;
    }
}