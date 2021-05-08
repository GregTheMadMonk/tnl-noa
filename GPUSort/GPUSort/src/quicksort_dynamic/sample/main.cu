#include <TNL/Containers/Array.h>
#include "../quicksort.cuh"
#include "../../util/algorithm.h"

#include <iostream>
#include <algorithm>
#include <numeric>
using namespace std;

int main()
{
    vector<int> vec(19);
    iota(vec.begin(), vec.end(), 0);
    random_shuffle(vec.begin(), vec.end());

    TNL::Containers::Array<int, TNL::Devices::Cuda> arr(vec);
    auto view = arr.getView();
    cout << view << endl;
    quicksort(view);
    cout << view << endl;

    return 0;
}