/***************************************************************************
                          BoundaryMeshFunctionTest.h  -  description
                             -------------------
    begin                : Aug 21, 2018
    copyright            : (C) 2018 by oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#ifdef HAVE_GTEST
#include <gtest/gtest.h>
#include <TNL/Functions/BoundaryMeshFunction.h>
#include <TNL/Meshes/Grid.h>

TEST( BoundaryMeshFunctionTest, BasicConstructor )
{
   using Grid = TNL::Meshes::Grid< 2 >;
   TNL::Functions::BoundaryMeshFunction< Grid > boundaryMesh;
}
#endif


#include "../main.h"
