#include <TNL/Containers/Array.h>
#include "quicksort.cuh"
#include "../util/algorithm.h"

#include <iostream>
#include <algorithm>
using namespace std;

int main()
{
    vector<int> vec(19);
    for(auto & x : vec) x = rand()%30;

    TNL::Containers::Array<int, TNL::Devices::Cuda> arr(vec);
    auto view = arr.getView();
    cout << view << endl;
    quicksort(view);
    cout << view << endl;

    return 0;
}