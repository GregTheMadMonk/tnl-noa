#include <iostream>
#include <TNL/Containers/Array.h>

#include "bitonicSort.h"
//--------------------------------------------------
std::ostream& operator<< (std::ostream&out, std::vector<int> &arr)
{
    for (auto x : arr)
        std::cout << x << " ";
    return std::cout << std::endl;
}

#define deb(x) std::cout << #x << " = " << x << std::endl;
//--------------------------------------------------

int main( int argc, char* argv[] )
{
    TNL::Containers::Array<int, Devices::Cuda> Arr(argc - 1);
    for(int i = 1; i < argc; i++)
        Arr.setElement(i-1, std::atoi(argv[i]));

    auto view = Arr.getView();

    std::cout << "unsorted: " << view << std::endl;
    bitonicSort(view);

    std::cout << "sorted: " << view << std::endl;

    return 0;
}