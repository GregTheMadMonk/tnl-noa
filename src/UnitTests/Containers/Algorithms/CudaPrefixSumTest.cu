/***************************************************************************
                          CudaPrefixSumTest.cu  -  description
                             -------------------
    begin                : Jun 6, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

/////
// NOTE: This test serves mainly for testing CUDA prefix sum when more than
// one CUDA grid is used. To avoid allocation of extremely large vectors and to
// speed-up this unit test, we decrease the grid size artificialy.

#pragma once

#ifdef HAVE_GTEST
#include <limits>

#include <TNL/Experimental/Arithmetics/Quad.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Containers/VectorView.h>
#include "VectorTestSetup.h"

#include "gtest/gtest.h"

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Containers::Algorithms;
using namespace TNL::Arithmetics;

