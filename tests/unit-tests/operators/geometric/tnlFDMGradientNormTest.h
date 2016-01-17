/***************************************************************************
                          tnlFDMGradientNormTest.h  -  description
                             -------------------
    begin                : Jan 17, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TNLFDMGRADIENTNORMTEST_H
#define	TNLFDMGRADIENTNORMTEST_H

#include <operators/geometric/tnlFDMGradientNorm.h>
#include <operators/geometric/tnlExactGradientNorm.h>

template< typename Mesh,
          typename Function,
          typename Operator,
          int MeshSize,
          bool Verbose >
void testDifferenceOperator()
{
   typedef tnlExactGradientNorm< Mesh::meshDimensions > ExactOperator;
   return tnlUnitTestStarter::run<
            tnlPDEOperatorEocTester< 
                Operator,
                ExactOperator,
                Function,
                MeshSize,
                Verbose > >();
   
}

template< typename Mesh,
          typename Function,
          int MeshSize,
          boolVerbose >
void setDifferenceOperator()
{
   typedef tnlFDMGradientNorm< Mesh, tnlForwardFiniteDifference > ForwardGradientNorm;
   typedef tnlFDMGradientNorm< Mesh, tnlBackwardFiniteDifference > BackwardGradientNorm;
   typedef tnlFDMGradientNorm< Mesh, tnlCentralFiniteDifference > CentralGradientNorm;
   return ( testDifferenceOperator< Mesh, Function, ForwardGradientNorm, MeshSize, Verboze >() &&
            testDifferenceOperator< Mesh, Function, BackwardGradientNorm, MeshSize, Verboze >() &&
            testDifferenceOperator< Mesh, Function, CentralGradientNorm, MeshSize, Verboze >() );
}

template< typename Mesh,
          int MeshSize,
          bool Verbose >
void setFunction()
{
   const int Dimensions = Mesh::meshDimensions;
   typedef tnlExpBumpFunction< Dimensions, RealType >  Function;
   return setDiferenceOperator< Mesh, Function, MeshSize, Verbose >();
}

template< int MeshSize,
          bool Verbose >
void setGrid()
{
   typedef tnlGrid< 1, MeshReal, Device, MeshIndex > Grid1D;
   typedef tnlGrid< 2, MeshReal, Device, MeshIndex > Grid2D;
   typedef tnlGrid< 3, MeshReal, Device, MeshIndex > Grid3D;
   return ( setFunction< Grid1D, MeshSize, Verbose >() &&
            setFunction< Grid2D, MeshSize, Verbose >() &&
            setFunction< Grid3D, MeshSize, Verbose >() );
}

int main( int argc, char* argv[] )
{
   const int meshSize( 32 );
   const bool verbose( false );
#ifdef HAVE_CPPUNIT
    return setGrid< MeshSize, Verbose >();
#else
   return EXIT_FAILURE;
#endif
}

#endif	/* TNLFDMGRADIENTNORMTEST_H */

