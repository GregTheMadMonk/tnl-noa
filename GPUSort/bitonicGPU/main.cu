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
    if(argc <= 1)
    {
        std::cout << "missing argument: N=array size to be tested on" << std::endl;
        return 1;
    }

    std::vector<int> a(std::atoi(argv[1]));
    for(int i = 0; i < a.size(); i++)
        a[i] = std::rand() % a.size();


    TNL::Containers::Array<int, TNL::Devices::Cuda> Arr(a);

    auto view = Arr.getView();

    
    //std::cout << "unsorted: " << view << std::endl;
    bitonicSort(a);

    //std::cout << "sorted: " << view << std::endl;

    return 0;
}