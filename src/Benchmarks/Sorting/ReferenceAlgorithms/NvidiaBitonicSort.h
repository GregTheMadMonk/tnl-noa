#include <6_Advanced/sortingNetworks/bitonicSort.cu>
#include <TNL/Containers/Array.h>


struct NvidiaBitonicSort
{
   static void sort( Containers::ArrayView< int, Devices::Cuda >& view )
   {
      Array<int, Devices::Cuda> arr;
      arr = view;
      bitonicSort((unsigned *)view.getData(), (unsigned *)arr.getData(),
                  (unsigned *)view.getData(), (unsigned *)arr.getData(),
                  1, arr.getSize(), 1);
      cudaDeviceSynchronize();
   }
};
