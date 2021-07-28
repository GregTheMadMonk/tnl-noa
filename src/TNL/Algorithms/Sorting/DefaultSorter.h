/***************************************************************************
                          DefaultSorter.h  -  description
                             -------------------
    begin                : Jul 14, 2021
    copyright            : (C) 2021 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Tomas Oberhuber

#pragma once

#include <TNL/Devices/Sequential.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Algorithms/Sorting/BitonicSort.h>
#include <TNL/Algorithms/Sorting/BubbleSort.h>
#include <TNL/Algorithms/Sorting/Quicksort.h>
#include <TNL/Algorithms/Sorting/STLSort.h>

namespace TNL {
   namespace Algorithms {
      namespace Sorting {

template< typename Device >
struct DefaultSorter;

template<>
struct DefaultSorter< Devices::Sequential >
{
   using SorterType = Algorithms::Sorting::STLSort;
};

template<>
struct DefaultSorter< Devices::Host >
{
   using SorterType = Algorithms::Sorting::STLSort;
};

template<>
struct DefaultSorter< Devices::Cuda >
{
   using SorterType = Algorithms::Sorting::Quicksort;
};

template< typename Device >
struct DefaultInplaceSorter;

template<>
struct DefaultInplaceSorter< Devices::Sequential >
{
   using SorterType = Algorithms::Sorting::BubbleSort;
};

template<>
struct DefaultInplaceSorter< Devices::Host >
{
   using SorterType = Algorithms::Sorting::BubbleSort;
};

template<>
struct DefaultInplaceSorter< Devices::Cuda >
{
   using SorterType = Algorithms::Sorting::BitonicSort;
};

      } // namespace Sorting
   } // namespace Algorithms
} // namespace TNL
