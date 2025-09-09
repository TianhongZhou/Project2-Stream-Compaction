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

        // Original Up-sweep Down-sweep
        //__global__ void upSweepKern(int n, int* odata, int d) {
        //    int k = (blockIdx.x * blockDim.x) + threadIdx.x;

        //    if (k >= n || k % (1 << (d + 1)) != 0) {
        //        return;
        //    }

        //    odata[k + (1 << (d + 1)) - 1] += odata[k + (1 << d) - 1];
        //}

        //__global__ void downSweepKern(int n, int* odata, int d) {
        //    int k = (blockIdx.x * blockDim.x) + threadIdx.x;

        //    if (k >= n || k % (1 << (d + 1)) != 0) {
        //        return;
        //    }

        //    int t = odata[k + (1 << d) - 1];
        //    odata[k + (1 << d) - 1] = odata[k + (1 << (d + 1)) - 1];
        //    odata[k + (1 << (d + 1)) - 1] += t;
        //}
        
        // Part 5 upgrade
        __global__ void upSweepKern(int n, int* data, int d) {
            int step = 1 << (d + 1);
            int k = blockIdx.x * blockDim.x + threadIdx.x;

            int right = (k + 1) * step - 1;
            if (right >= n) {
                return;
            }

            int left = right - (step >> 1);
            data[right] += data[left];
        }

        __global__ void downSweepKern(int n, int* data, int d) {
            int step = 1 << (d + 1);
            int k = blockIdx.x * blockDim.x + threadIdx.x;

            int right = (k + 1) * step - 1;
            if (right >= n) {
                return;
            }

            int left = right - (step >> 1);
            int t = data[left];
            data[left] = data[right];
            data[right] += t;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata, bool time) {
            if (n <= 0) {
                return;
            }
            int logn = ilog2ceil(n);
            int new_n = 1 << logn;

            int* dev_buffer = nullptr;
            cudaMalloc(&dev_buffer, new_n * sizeof(int));
            cudaMemset(dev_buffer, 0, new_n * sizeof(int));
            cudaMemcpy(dev_buffer, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            const int blockSize = 128;

            if (time) timer().startGpuTimer();

            for (int d = 0; d <= logn - 1; ++d) {
                int step = 1 << (d + 1);
                int thr = new_n / step;
                dim3 grid = dim3((thr + blockSize - 1) / blockSize);
                upSweepKern<<<grid, blockSize>>>(new_n, dev_buffer, d);
            }

            cudaMemset(dev_buffer + (new_n - 1), 0, sizeof(int));

            for (int d = logn - 1; d >= 0; --d) {
                int step = 1 << (d + 1);
                int thr = new_n / step;
                dim3 grid = dim3((thr + blockSize - 1) / blockSize);
                downSweepKern<<<grid, blockSize>>>(new_n, dev_buffer, d);
            }

            cudaDeviceSynchronize();
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
