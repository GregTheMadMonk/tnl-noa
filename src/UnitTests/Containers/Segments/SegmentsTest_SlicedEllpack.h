/***************************************************************************
                          SegmentsTest_SlicedEllpack.h -  description
                             -------------------
    begin                : Dec 9, 2019
    copyright            : (C) 2019 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/Algorithms/Segments/SlicedEllpack.h>

#include "SegmentsTest.hpp"
#include <iostream>

#ifdef HAVE_GTEST
#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Segments >
class SlicedEllpackSegmentsTest : public ::testing::Test
{
protected:
   using SlicedEllpackSegmentsType = Segments;
};

// types for which MatrixTest is instantiated
using SlicedEllpackSegmentsTypes = ::testing::Types
<
    TNL::Algorithms::Segments::SlicedEllpack< TNL::Devices::Host, int    >,
    TNL::Algorithms::Segments::SlicedEllpack< TNL::Devices::Host, long   >
#ifdef HAVE_CUDA
   ,TNL::Algorithms::Segments::SlicedEllpack< TNL::Devices::Cuda, int    >,
    TNL::Algorithms::Segments::SlicedEllpack< TNL::Devices::Cuda, long   >
#endif
>;

TYPED_TEST_SUITE( SlicedEllpackSegmentsTest, SlicedEllpackSegmentsTypes );

TYPED_TEST( SlicedEllpackSegmentsTest, setSegmentsSizes_EqualSizes )
{
    using SlicedEllpackSegmentsType = typename TestFixture::SlicedEllpackSegmentsType;

    test_SetSegmentsSizes_EqualSizes< SlicedEllpackSegmentsType >();
}

TYPED_TEST( SlicedEllpackSegmentsTest, allReduction_MaximumInSegments )
{
    using SlicedEllpackSegmentsType = typename TestFixture::SlicedEllpackSegmentsType;

    test_AllReduction_MaximumInSegments< SlicedEllpackSegmentsType >();
}

#endif

#include "../../main.h"
