/***************************************************************************
                          VectorTestSetup.h  -  description
                             -------------------
    begin                : Apr 30, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#ifdef HAVE_GTEST
#include <limits>

#include <TNL/Experimental/Arithmetics/Quad.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Containers/VectorView.h>
#include "VectorHelperFunctions.h"

#include "gtest/gtest.h"

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Arithmetics;

// test fixture for typed tests
template< typename Vector >
class VectorTest : public ::testing::Test
{
protected:
   using VectorType = Vector;
   using ViewType = VectorView< typename Vector::RealType, typename Vector::DeviceType, typename Vector::IndexType >;
};

// types for which VectorTest is instantiated
// TODO: Quad must be fixed
using VectorTypes = ::testing::Types<
#ifndef HAVE_CUDA
   Vector< int,            Devices::Host, short >,
   Vector< long,           Devices::Host, short >,
   Vector< float,          Devices::Host, short >,
   Vector< double,         Devices::Host, short >,
   //Vector< Quad< float >,  Devices::Host, short >,
   //Vector< Quad< double >, Devices::Host, short >,
   Vector< int,            Devices::Host, int >,
   Vector< long,           Devices::Host, int >,
   Vector< float,          Devices::Host, int >,
   Vector< double,         Devices::Host, int >,
   //Vector< Quad< float >,  Devices::Host, int >,
   //Vector< Quad< double >, Devices::Host, int >,
   Vector< int,            Devices::Host, long >,
   Vector< long,           Devices::Host, long >,
   Vector< float,          Devices::Host, long >,
   Vector< double,         Devices::Host, long >
   //Vector< Quad< float >,  Devices::Host, long >,
   //Vector< Quad< double >, Devices::Host, long >
#endif
#ifdef HAVE_CUDA
   Vector< int,            Devices::Cuda, short >,
   Vector< long,           Devices::Cuda, short >,
   Vector< float,          Devices::Cuda, short >,
   Vector< double,         Devices::Cuda, short >,
   //Vector< Quad< float >,  Devices::Cuda, short >,
   //Vector< Quad< double >, Devices::Cuda, short >,
   Vector< int,            Devices::Cuda, int >,
   Vector< long,           Devices::Cuda, int >,
   Vector< float,          Devices::Cuda, int >,
   Vector< double,         Devices::Cuda, int >,
   //Vector< Quad< float >,  Devices::Cuda, int >,
   //Vector< Quad< double >, Devices::Cuda, int >,
   Vector< int,            Devices::Cuda, long >,
   Vector< long,           Devices::Cuda, long >,
   Vector< float,          Devices::Cuda, long >,
   Vector< double,         Devices::Cuda, long >
   //Vector< Quad< float >,  Devices::Cuda, long >,
   //Vector< Quad< double >, Devices::Cuda, long >
#endif
#ifdef HAVE_MIC
   ,
   Vector< int,    Devices::MIC, short >,
   Vector< long,   Devices::MIC, short >,
   Vector< float,  Devices::MIC, short >,
   Vector< double, Devices::MIC, short >,
   Vector< int,    Devices::MIC, int >,
   Vector< long,   Devices::MIC, int >,
   Vector< float,  Devices::MIC, int >,
   Vector< double, Devices::MIC, int >,
   Vector< int,    Devices::MIC, long >,
   Vector< long,   Devices::MIC, long >,
   Vector< float,  Devices::MIC, long >,
   Vector< double, Devices::MIC, long >
#endif
>;

TYPED_TEST_SUITE( VectorTest, VectorTypes );
#endif
