/***************************************************************************
                          SegmentsTest_Ellpack.h -  description
                             -------------------
    begin                : Dec 6, 2019
    copyright            : (C) 2019 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/Containers/Segments/Ellpack.h>

#include "SegmentsTest.hpp"
#include <iostream>

#ifdef HAVE_GTEST
#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Segments >
class EllpackSegmentsTest : public ::testing::Test
{
protected:
   using EllpackSegmentsType = Segments;
};

// types for which MatrixTest is instantiated
using EllpackSegmentsTypes = ::testing::Types
<
    TNL::Containers::Segments::Ellpack< TNL::Devices::Host, int    >,
    TNL::Containers::Segments::Ellpack< TNL::Devices::Host, long   >,
    TNL::Containers::Segments::Ellpack< TNL::Devices::Host, int    >,
    TNL::Containers::Segments::Ellpack< TNL::Devices::Host, long   >,
    TNL::Containers::Segments::Ellpack< TNL::Devices::Host, int    >,
    TNL::Containers::Segments::Ellpack< TNL::Devices::Host, long   >
#ifdef HAVE_CUDA
   ,TNL::Containers::Segments::Ellpack< TNL::Devices::Cuda, int    >,
    TNL::Containers::Segments::Ellpack< TNL::Devices::Cuda, long   >,
    TNL::Containers::Segments::Ellpack< TNL::Devices::Cuda, int    >,
    TNL::Containers::Segments::Ellpack< TNL::Devices::Cuda, long   >,
    TNL::Containers::Segments::Ellpack< TNL::Devices::Cuda, int    >,
    TNL::Containers::Segments::Ellpack< TNL::Devices::Cuda, long   >,
#endif
>;

TYPED_TEST_SUITE( EllpackSegmentsTest, EllpackSegmentsTypes );

TYPED_TEST( EllpackSegmentsTest, setSegmentsSizes_EqualSizes )
{
    using EllpackSegmentsType = typename TestFixture::EllpackSegmentsType;

    test_SetSegmentsSizes_EqualSizes< EllpackSegmentsType >();
}

TYPED_TEST( EllpackSegmentsTest, setSegmentsSizes_EqualSizes_EllpackOnly )
{
    using EllpackSegmentsType = typename TestFixture::EllpackSegmentsType;

    test_SetSegmentsSizes_EqualSizes_EllpackOnly< EllpackSegmentsType >();
}

TYPED_TEST( EllpackSegmentsTest, getMaxInSegments )
{
    using EllpackSegmentsType = typename TestFixture::EllpackSegmentsType;

    test_GetMaxInSegments< EllpackSegmentsType >();
}

#endif

#include "../../main.h"
