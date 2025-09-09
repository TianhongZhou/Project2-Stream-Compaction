#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void upSweepKern(int n, int* odata, int d) {
            int k = (blockIdx.x * blockDim.x) + threadIdx.x;

            if (k >= n || k % (1 << (d + 1)) != 0) {
                return;
            }

            odata[k + (1 << (d + 1)) - 1] += odata[k + (1 << d) - 1];
        }

        __global__ void downSweepKern(int n, int* odata, int d) {
            int k = (blockIdx.x * blockDim.x) + threadIdx.x;

            if (k >= n || k % (1 << (d + 1)) != 0) {
                return;
            }

            int t = odata[k + (1 << d) - 1];
            odata[k + (1 << d) - 1] = odata[k + (1 << (d + 1)) - 1];
            odata[k + (1 << (d + 1)) - 1] += t;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int logn = ilog2ceil(n);
            int new_n = 1 << logn;

            int* dev_buffer;

            cudaMalloc((void**)&dev_buffer, new_n * sizeof(int));
            cudaMemset(dev_buffer, 0, new_n * sizeof(int));
            cudaMemcpy(dev_buffer, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int blockSize = 128;
            dim3 blockNum((new_n + blockSize - 1) / blockSize);

            timer().startGpuTimer();
            // TODO
            // Up-Sweep
            for (int d = 0; d <= logn - 1; d++) {
                upSweepKern<<<blockNum, blockSize>>>(new_n, dev_buffer, d);
            }

            cudaMemset(&dev_buffer[new_n - 1], 0, sizeof(int));

            // Down-Sweep
            for (int d = logn - 1; d >= 0; d--) {
                downSweepKern<<<blockNum, blockSize >>>(new_n, dev_buffer, d);
            }

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_buffer, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_buffer);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            timer().endGpuTimer();
            return -1;
        }
    }
}
