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
        void scan(int n, int *odata, const int *idata, bool time) {
            int logn = ilog2ceil(n);
            int new_n = 1 << logn;

            int* dev_buffer;

            cudaMalloc((void**)&dev_buffer, new_n * sizeof(int));
            cudaMemset(dev_buffer, 0, new_n * sizeof(int));
            cudaMemcpy(dev_buffer, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int blockSize = 128;
            dim3 blockNum((new_n + blockSize - 1) / blockSize);

            if (time) timer().startGpuTimer();
            // TODO
            // Up-Sweep
            for (int d = 0; d <= logn - 1; d++) {
                upSweepKern<<<blockNum, blockSize>>>(new_n, dev_buffer, d);
                cudaDeviceSynchronize();
            }

            cudaMemset(&dev_buffer[new_n - 1], 0, sizeof(int));

            // Down-Sweep
            for (int d = logn - 1; d >= 0; d--) {
                downSweepKern<<<blockNum, blockSize>>>(new_n, dev_buffer, d);
                cudaDeviceSynchronize();
            }

            if (time) timer().endGpuTimer();

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
            int* dev_idata;
            int* dev_odata;
            int* dev_bool;
            int* dev_indicies;

            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            cudaMalloc((void**)&dev_bool, n * sizeof(int));
            cudaMalloc((void**)&dev_indicies, n * sizeof(int));

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int blockSize = 128;
            dim3 blockNum((n + blockSize - 1) / blockSize);

            timer().startGpuTimer();
            // TODO
            StreamCompaction::Common::kernMapToBoolean<<<blockNum, blockSize>>>(n, dev_bool, dev_idata);
            cudaDeviceSynchronize();

            scan(n, dev_indicies, dev_bool, false);
            cudaDeviceSynchronize();

            StreamCompaction::Common::kernScatter<<<blockNum, blockSize>>>(n, dev_odata, dev_idata, dev_bool, dev_indicies);
            cudaDeviceSynchronize();

            timer().endGpuTimer();

            int lastBool;
            int lastIndex;

            cudaMemcpy(&lastBool, dev_bool + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastIndex, dev_indicies + n - 1, sizeof(int), cudaMemcpyDeviceToHost);

            int count = lastBool + lastIndex;
            cudaMemcpy(odata, dev_odata, count * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_odata);
            cudaFree(dev_idata);
            cudaFree(dev_bool);
            cudaFree(dev_indicies);

            return count;
        }
    }
}
