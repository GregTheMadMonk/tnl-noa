#include <iostream>

using namespace std;


#if defined(HAVE_MPI) && defined(HAVE_CUDA)

#include <cuda_runtime.h>
#include <mpi.h>
 
__global__ void SetKernel(float *deviceData, float value)
{
    // Just a dummy kernel
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    deviceData[idx] = value;
}

double sum(float * data, int count)
{
    double sum=0;
    for(int i=0;i<count;i++)
        sum+=data[i];

    return sum;
}


int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int blockSize = 256;
    int gridSize = 1000;

    int dataCount=blockSize*gridSize;

    float * deviceData=NULL;
    cudaMalloc((void **)&deviceData, dataCount * sizeof(float));

    if(rank==0)
    {
        cout << rank<<": "<<"Setup GPU alocated array to 1" << endl;
        SetKernel<<< gridSize,blockSize >>>(deviceData,1.0f);
        cout << rank<<": "<<" Sending GPU data " <<endl;
        MPI_Send((void*)deviceData, dataCount, MPI_FLOAT, 1, 1, MPI_COMM_WORLD);
    }
    
    if(rank==1) 
    {
        cout << rank<<": "<<" Reciving GPU data " <<endl;
        MPI_Recv((void*) deviceData, dataCount, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        float *data = new float[dataCount];
        cout << rank<<": "<<" Copying data from GPU to CPU " <<endl;
        cudaMemcpy( (void*) data, (void*)deviceData, dataCount*sizeof(float),  cudaMemcpyDeviceToHost);    
        cout << rank<<": "<<" Computin Sum on CPU " <<endl;
        cout << rank<<": "<< "sum:" << sum(data,dataCount) << endl;
        delete [] data;
    }

    cudaFree(deviceData);

    MPI_Finalize();
return 0;
}

#else

int main(void)
{
    cout << "CUDA or MPI missing...." <<endl;
}

#endif

 
