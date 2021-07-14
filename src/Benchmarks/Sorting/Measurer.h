#pragma once

#include <vector>
#include <TNL/Devices/Cuda.h>
#include <TNL/Containers/Array.h>
#include <TNL/Algorithms/Sort.h>

#ifdef HAVE_CUDA
#include "ReferenceAlgorithms/manca_quicksort.h"
#include "ReferenceAlgorithms/cederman_qsort.h"
#endif

#include "timer.h"

using namespace TNL;

struct QuicksortSorter
{
    template< typename Array >
    static void sort( Array& array ) {
       Algorithms::Sorting::Quicksort::sort( array );
    };
};

struct BitonicSortSorter
{
    template< typename Array >
    static void sort( Array& array ) { Algorithms::detail::bitonicSort( array ); };
};

struct STLSorter
{
    template< typename Value >
    static void sort( std::vector< Value >& vec ) { std::sort( vec.begin(), vec.end() ); };
};

#ifdef HAVE_CUDA
struct MancaQuicksortSorter
{
   static void sort( Containers::ArrayView< int, Devices::Cuda >& array )
   {
      double timer;
      CUDA_Quicksort( ( unsigned * ) array.getData(),  (unsigned * ) array.getData(), array.getSize(), 256, 0, &timer );
      //return;
   }
};

struct CedermanQuicksortSorter
{
   static void sort( Containers::ArrayView< int, Devices::Cuda >& array )
   {
      gpuqsort( ( unsigned int * ) array.getData(), ( unsigned int ) array.getSize() );
   }
};
#endif


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
struct Measurer< STLSorter >
{
   template< typename Value >
   static double measure( const std::vector<Value>&vec, int tries, int & wrongAnsCnt )
   {
      vector<double> resAcc;

      for(int i = 0; i < tries; i++)
      {
         std::vector< Value > vec2 = vec;
         {
               TIMER t([&](double res){resAcc.push_back(res);});
               STLSorter::sort( vec2 );
         }
      }
      return accumulate(resAcc.begin(), resAcc.end(), 0.0) / resAcc.size();
   }
};
