#include <string>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <set>
#include <iomanip>

#include <TNL/Containers/Array.h>

#include "../bitonicSort.h"
#include "../../util/timer.h"


using namespace TNL;
using namespace TNL::Containers;

typedef Devices::Cuda Device;

using namespace std;
int main()
{

    for(int pow = 10; pow <= 20; pow++)
    {
        int size =(1<< pow);

        vector<int> vec(size);
        iota(vec.begin(), vec.end(), 0);

        Array<int, Device> arr;
        vector<double> resAcc;

        //sorted sequence
        {
            arr = vec;
            auto view = arr.getView();

            {
                TIMER t([&](double res){resAcc.push_back(res);});
                bitonicSort(view);
            }
        }

        //almost sorted sequence
        {
            for(int i = 0; i < 3; i++)
            {
                int s = std::rand() % (size - 3);
                std::swap(vec[s], vec[s + 1]);
            }

            arr = vec;
            auto view = arr.getView();

            {
                TIMER t([&](double res){resAcc.push_back(res);});
                bitonicSort(view);
            }
        }

        //decreasing sequence
        {
            for(size_t i = 0; i < size; i++)
                vec[i] = -i;
                
            arr = vec;
            auto view = arr.getView();

            {
                TIMER t([&](double res){resAcc.push_back(res);});
                bitonicSort(view);
            }
        }
        
        //random sequence
        {
            std::random_shuffle(vec.begin(), vec.end());

            arr = vec;
            auto view = arr.getView();
            
            {
                TIMER t([&](double res){resAcc.push_back(res);});
                bitonicSort(view);
            }
        }


        cout << "2^" << pow << " = ";
        cout << fixed;
        cout << setprecision(3);
        cout << (accumulate(resAcc.begin(), resAcc.end(), 0.0) / resAcc.size()) << " ms" << endl;
    }

    return 0;
}