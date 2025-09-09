#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "radix.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Radix {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernBitFlag(int n, int* bits, const int* data, int bit) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= n) {
                return;
            }

            bits[i] = (data[i] >> bit) & 1;
        }

        __global__ void kernInvert01(int n, int* outZeros, const int* inBits) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= n) {
                return;
            }

            outZeros[i] = 1 - inBits[i];   
        }

        __global__ void kernScatterRadix(int n, int* out, const int* in, const int* bits, const int* zerosScan, int totalZeros) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= n) {
                return;
            }

            int b = bits[i];
            int pos;
            if (b == 0) {
                pos = zerosScan[i];
            }
            else {
                pos = totalZeros + (i - zerosScan[i]);
            }
            out[pos] = in[i];
        }

        void sort(int n, int* odata, const int* idata) {
            if (n <= 0) {
                return;
            }

            int* dev_idata;
            int* dev_odata;
            int* dev_bits;
            int* dev_zeros;
            int* dev_scan;

            cudaMalloc(&dev_idata, n * sizeof(int));
            cudaMalloc(&dev_odata, n * sizeof(int));
            cudaMalloc(&dev_bits, n * sizeof(int));
            cudaMalloc(&dev_zeros, n * sizeof(int));
            cudaMalloc(&dev_scan, n * sizeof(int));

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int blockSize = 128;
            dim3 blockNum((n + blockSize - 1) / blockSize);

            timer().startGpuTimer();

            for (int bit = 0; bit < 32; ++bit) {
                kernBitFlag<<<blockNum, blockSize>>>(n, dev_bits, dev_idata, bit);
                kernInvert01<<<blockNum, blockSize>>>(n, dev_zeros, dev_bits);
                Efficient::scan(n, dev_scan, dev_zeros, false);

                int host_lastScan = 0, host_lastZero = 0;
                cudaMemcpy(&host_lastScan, dev_scan + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(&host_lastZero, dev_zeros + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
                int totalZeros = host_lastScan + host_lastZero;

                kernScatterRadix<<<blockNum, blockSize>>>(n, dev_odata, dev_idata, dev_bits, dev_scan, totalZeros);
                std::swap(dev_idata, dev_odata);
            }

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_scan);
            cudaFree(dev_zeros);
            cudaFree(dev_bits);
            cudaFree(dev_odata);
            cudaFree(dev_idata);
        }
    }
}