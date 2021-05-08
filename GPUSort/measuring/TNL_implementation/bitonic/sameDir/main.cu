#include "../../../GPUSort/bitonicGPU/bitonicSort.h"
#include <TNL/Algorithms/MemoryOperations.h>
#include "../../util/timer.h"
#include "../../util/algorithm.h"

#include <iostream>
#include <algorithm>
#include <numeric>
#include <vector>
#include <set>
#include <random>
#include <fstream>


class NOT_SORTED_PROPERLY{};

using namespace std;
int main()
{
    ofstream out("TNL_sameDir.csv");
    out << "implementation,size,sorted,almost_sorted,decreasing,random" << endl; 

    for(int pow = 3; pow <= 23 ; pow++)
    {
        int size =(1<< pow);
        std::set<int> sizes{size, size+1, size-1};
        for(int i = 0; i < 3; i++)
            sizes.insert(size + (std::rand() % size));

        for(auto x : sizes)
        {
            cout << "checking size =" << x << endl;

            out << "TNL," << x;
            std::vector<int> vec(x);
            for(int i = 0; i < x ; ++i)
                vec[i] = i;
            TNL::Containers::Array<int, TNL::Devices::Cuda> arr;
            
            //sorted sequence
            {
                arr = vec;
                auto view = arr.getView();
                {
                    TIMER t([&](double res){out << "," << res;});
                    bitonicSort(arr.getView());
                }

                if(!is_sorted(arr.getView()))
                {
                    cerr << "sorted seq" << endl;
                    throw NOT_SORTED_PROPERLY();
                }
            }

            //almost sorted sequence
            {
                for(int i = 0; i < 3; i++)
                {
                    int s = std::rand() % (x - 3);
                    std::swap(vec[s], vec[s + 1]);
                }

                auto view = arr.getView();
                {
                    TIMER t([&](double res){out << "," << res;});
                    bitonicSort(arr.getView());
                }

                if(!is_sorted(arr.getView()))
                {
                    cerr << "almost sorted seq" << endl;
                    throw NOT_SORTED_PROPERLY();
                }
            }

            //decreasing sequence
            {
                for(size_t i = 0; i < x; i++)
                    vec[i] = -i;
                    
                auto view = arr.getView();
                {
                    TIMER t([&](double res){out << "," << res;});
                    bitonicSort(arr.getView());
                }

                if(!is_sorted(arr.getView()))
                {
                    cerr << "dec seq" << endl;
                    throw NOT_SORTED_PROPERLY();
                }
            }
            
            //random sequence
            {
                std::random_shuffle(vec.begin(), vec.end());

                auto view = arr.getView();
                {
                    TIMER t([&](double res){out << "," << res;});
                    bitonicSort(arr.getView());
                }

                if(!is_sorted(arr.getView()))
                {
                    cerr << "random seq" << endl;
                    throw NOT_SORTED_PROPERLY();
                }
            }

            out << endl;
        }

    }

    return 0;
}