#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void scanKern(int n, int* odata, const int* idata, int pow2dm1) {
            int k = (blockIdx.x * blockDim.x) + threadIdx.x;

            if (k >= n) {
                return;
            }

            odata[k] = k >= pow2dm1 ? idata[k - pow2dm1] + idata[k] : idata[k];
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int logn = ilog2ceil(n);

            int* dev_bufferA;
            int* dev_bufferB;

            cudaMalloc((void**)&dev_bufferA, n * sizeof(int));
            cudaMalloc((void**)&dev_bufferB, n * sizeof(int));
            cudaMemcpy(dev_bufferA, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int blockSize = 128;
            dim3 blockNum((n + blockSize - 1) / blockSize);

            int* read;
            int* write;

            timer().startGpuTimer();
            // TODO
            for (int d = 1; d <= logn; d++) {
                read = d % 2 == 1 ? dev_bufferA : dev_bufferB;
                write = d % 2 == 1 ? dev_bufferB : dev_bufferA;

                scanKern<<<blockNum, blockSize>>>(n, write, read, 1 << (d - 1));
            }
            timer().endGpuTimer();

            cudaMemcpy(odata, write, n * sizeof(int), cudaMemcpyDeviceToHost);

            for (int i = n - 1; i > 0; i--) {
                odata[i] = odata[i - 1];
            }
            odata[0] = 0;

            cudaFree(dev_bufferA);
            cudaFree(dev_bufferB);
        }
    }
}
