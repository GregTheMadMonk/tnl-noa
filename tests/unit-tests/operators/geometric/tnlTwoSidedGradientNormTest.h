/***************************************************************************
                          tnlTwoSidedGradientNormTest.h  -  description
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

#ifndef TNLTWOSIDEDGRADIENTNORMTEST_H
#define	TNLTWOSIDEDGRADIENTNORMTEST_H

#include <operators/geometric/tnlTwoSidedGradientNorm.h>
#include <operators/geometric/tnlExactGradientNorm.h>
#include "../../tnlUnitTestStarter.h"
#include "../tnlPDEOperatorEocTester.h"

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename TestFunction >
class tnlPDEOperatorEocTestResult< 
   tnlTwoSidedGradientNorm< tnlGrid< Dimensions, Real, Device, Index >,
                            Real,
                            Index >,
   TestFunction >
{
   // TODO: Check this - EOC for 1D is only 1.5
   public:
      static Real getL1Eoc() { return ( Real ) 2.0; };
      static Real getL1Tolerance() { return ( Real ) 0.5; };

      static Real getL2Eoc() { return ( Real ) 2.0; };
      static Real getL2Tolerance() { return ( Real ) 0.6; };

      static Real getMaxEoc() { return ( Real ) 1.0; };
      static Real getMaxTolerance() { return ( Real ) 0.5; };

};

template< typename Mesh,
          typename Function,
          typename Operator,
          int MeshSize,
          bool WriteFunctions,
          bool Verbose >
bool testDifferenceOperator()
{
   typedef tnlExactGradientNorm< Mesh::meshDimensions > ExactOperator;
   return tnlUnitTestStarter::run<
            tnlPDEOperatorEocTester< 
                Operator,
                ExactOperator,
                Function,
                typename Mesh::Cell,
                MeshSize,
                WriteFunctions,
                Verbose > >();
   
}

template< typename Mesh,
          typename Function,
          int MeshSize,
          bool WriteFunctions,
          bool Verbose >
bool setDifferenceOperator()
{
   typedef tnlTwoSidedGradientNorm< Mesh > GradientNorm;
   return ( testDifferenceOperator< Mesh, Function, GradientNorm, MeshSize, WriteFunctions, Verbose >() );
}

template< typename Mesh,
          int MeshSize,
          bool WriteFunctions,
          bool Verbose >
bool setFunction()
{
   const int Dimensions = Mesh::meshDimensions;
   typedef tnlExpBumpFunction< Dimensions, double >  Function;
   return setDifferenceOperator< Mesh, Function, MeshSize, WriteFunctions, Verbose >();
}

template< typename Device,
          //int MeshSize,
          bool WriteFunctions,
          bool Verbose >
bool setGrid()
{
   typedef double MeshReal;
   typedef int MeshIndex;
   typedef tnlGrid< 1, MeshReal, Device, MeshIndex > Grid1D;
   typedef tnlGrid< 2, MeshReal, Device, MeshIndex > Grid2D;
   typedef tnlGrid< 3, MeshReal, Device, MeshIndex > Grid3D;
   return ( setFunction< Grid1D, 65536, WriteFunctions, Verbose >() &&
            setFunction< Grid2D, 512, WriteFunctions, Verbose >() &&
            setFunction< Grid3D, 32, WriteFunctions, Verbose >() );
}

int main( int argc, char* argv[] )
{
   //const int meshSize( 32 );
   const bool writeFunctions( true );
   const bool verbose( true );
#ifdef HAVE_CPPUNIT
    return setGrid< tnlHost, writeFunctions, verbose >();
#else
   return EXIT_FAILURE;
#endif
}

#endif	/* TNLFDMGRADIENTNORMTEST_H */

