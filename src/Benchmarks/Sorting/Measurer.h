#pragma once

#include <vector>
#include <TNL/Devices/Cuda.h>
#include <TNL/Containers/Array.h>
#include <TNL/Algorithms/Sorting/Quicksort.h>
#include <TNL/Algorithms/Sorting/BitonicSort.h>
#include <TNL/Algorithms/Sorting/STLSort.h>

#ifdef HAVE_CUDA
#include "ReferenceAlgorithms/MancaQuicksort.h"
#include "ReferenceAlgorithms/CedermanQuicksort.h"
#endif

#include "timer.h"

using namespace TNL;

/*struct STLSorter
{
    template< typename Value >
    static void sort( std::vector< Value >& vec ) { std::sort( vec.begin(), vec.end() ); };
};*/


template< typename Sorter >
struct Measurer
{
   template< typename Value >
   static double measure( const std::vector<Value>&vec, int tries, int & wrongAnsCnt )
   {
      vector<double> resAcc;

      for(int i = 0; i < tries; i++)
      {
         Containers::Array<Value, Devices::Cuda > arr(vec);
         auto view = arr.getView();
         {
               TIMER t([&](double res){resAcc.push_back(res);});
               Sorter::sort(view);
         }

         if( ! Algorithms::isSorted( view ) )
               wrongAnsCnt++;
      }
      return accumulate(resAcc.begin(), resAcc.end(), 0.0) / resAcc.size();
   }
};

template<>
struct Measurer< Algorithms::Sorting::STLSort >
{
   template< typename Value >
   static double measure( const std::vector<Value>&vec, int tries, int & wrongAnsCnt )
   {
      vector<double> resAcc;

      for(int i = 0; i < tries; i++)
      {
         Containers::Array<Value, Devices::Host > arr(vec);
         auto view = arr.getView();
         //std::vector< Value > vec2 = vec;
         {
               TIMER t([&](double res){resAcc.push_back(res);});
               Algorithms::Sorting::STLSort::sort( view );
         }
      }
      return accumulate(resAcc.begin(), resAcc.end(), 0.0) / resAcc.size();
   }
};
