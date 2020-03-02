/***************************************************************************
                          SegmentsTest_CSR.h -  description
                             -------------------
    begin                : Nov 2, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/Containers/Segments/CSR.h>

#include "SegmentsTest.hpp"
#include <iostream>

#ifdef HAVE_GTEST
#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Segments >
class CSRSegmentsTest : public ::testing::Test
{
protected:
   using CSRSegmentsType = Segments;
};

// types for which MatrixTest is instantiated
using CSRSegmentsTypes = ::testing::Types
<
    TNL::Containers::Segments::CSR< TNL::Devices::Host, int    >,
    TNL::Containers::Segments::CSR< TNL::Devices::Host, long   >,
    TNL::Containers::Segments::CSR< TNL::Devices::Host, int    >,
    TNL::Containers::Segments::CSR< TNL::Devices::Host, long   >,
    TNL::Containers::Segments::CSR< TNL::Devices::Host, int    >,
    TNL::Containers::Segments::CSR< TNL::Devices::Host, long   >
#ifdef HAVE_CUDA
   ,TNL::Containers::Segments::CSR< TNL::Devices::Cuda, int    >,
    TNL::Containers::Segments::CSR< TNL::Devices::Cuda, long   >,
    TNL::Containers::Segments::CSR< TNL::Devices::Cuda, int    >,
    TNL::Containers::Segments::CSR< TNL::Devices::Cuda, long   >,
    TNL::Containers::Segments::CSR< TNL::Devices::Cuda, int    >,
    TNL::Containers::Segments::CSR< TNL::Devices::Cuda, long   >,
#endif
>;

TYPED_TEST_SUITE( CSRSegmentsTest, CSRSegmentsTypes );

TYPED_TEST( CSRSegmentsTest, setSegmentsSizes_EqualSizes )
{
    using CSRSegmentsType = typename TestFixture::CSRSegmentsType;

    test_SetSegmentsSizes_EqualSizes< CSRSegmentsType >();
}

#endif

#include "../../main.h"
