/***************************************************************************
                          tnlFiniteDifferencesTest.h  -  description
                             -------------------
    begin                : Jan 12, 2016
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

#include <tnlConfig.h>
#include <core/tnlHost.h>
#include <cstdlib>

#include "../tnlPDEOperatorEocTester.h"
#include "../../tnlUnitTestStarter.h"
#include <mesh/tnlGrid.h>
#include <operators/fdm/tnlBackwardFiniteDifference.h>
#include <operators/fdm/tnlForwardFiniteDifference.h>
#include <operators/fdm/tnlCentralFiniteDifference.h>
#include <operators/fdm/tnlExactDifference.h>
#include "../tnlPDEOperatorEocTestResult.h"
#include <functions/tnlExpBumpFunction.h>

template< typename MeshType,
          typename RealType,
          int XDifference,
          int YDifference,
          int ZDifference,
          typename IndexType,   
          typename TestFunction >
class tnlPDEOperatorEocTestResult< 
   tnlForwardFiniteDifference< MeshType, XDifference, YDifference, ZDifference, RealType, IndexType >, TestFunction >
{
   public:
      static Real getL1Eoc() 
      {
         if( XDifference < 2 && YDifference < 2 && ZDifference < 2 )
            return ( Real ) 1.0;
      };
      static Real getL1Tolerance() { return ( Real ) 0.05; };

      static Real getL2Eoc()
      { 
         if( XDifference < 2 && YDifference < 2 && ZDifference < 2 )
            return ( Real ) 1.0;
      };
      static Real getL2Tolerance() { return ( Real ) 0.05; };

      static Real getMaxEoc()
      {
         if( XDifference < 2 && YDifference < 2 && ZDifference < 2 )
            return ( Real ) 1.0; 
      };
      static Real getMaxTolerance() { return ( Real ) 0.05; };
};

template< typename MeshType,
          typename RealType,
          int XDifference,
          int YDifference,
          int ZDifference,
          typename IndexType,   
          typename TestFunction >
class tnlPDEOperatorEocTestResult< 
   tnlBackwardFiniteDifference< MeshType, XDifference, YDifference, ZDifference, RealType, IndexType >, TestFunction >
{
   public:
      static Real getL1Eoc() 
      {
         if( XDifference < 2 && YDifference < 2 && ZDifference < 2 )
            return ( Real ) 1.0;
      };
      static Real getL1Tolerance() { return ( Real ) 0.05; };

      static Real getL2Eoc()
      { 
         if( XDifference < 2 && YDifference < 2 && ZDifference < 2 )
            return ( Real ) 1.0;
      };
      static Real getL2Tolerance() { return ( Real ) 0.05; };

      static Real getMaxEoc()
      {
         if( XDifference < 2 && YDifference < 2 && ZDifference < 2 )
            return ( Real ) 1.0; 
      };
      static Real getMaxTolerance() { return ( Real ) 0.05; };

};

template< typename MeshType,
          typename RealType,
          int XDifference,
          int YDifference,
          int ZDifference,
          typename IndexType,   
          typename TestFunction >
class tnlPDEOperatorEocTestResult< 
   tnlCentralFiniteDifference< MeshType, XDifference, YDifference, ZDifference, RealType, IndexType >, TestFunction >
{
   public:
      static Real getL1Eoc() 
      {
         if( XDifference < 2 && YDifference < 2 && ZDifference < 2 )
            return ( Real ) 2.0;
      };
      static Real getL1Tolerance() { return ( Real ) 0.05; };

      static Real getL2Eoc()
      { 
         if( XDifference < 2 && YDifference < 2 && ZDifference < 2 )
            return ( Real ) 2.0;
      };
      static Real getL2Tolerance() { return ( Real ) 0.05; };

      static Real getMaxEoc()
      {
         if( XDifference < 2 && YDifference < 2 && ZDifference < 2 )
            return ( Real ) 2.0; 
      };
      static Real getMaxTolerance() { return ( Real ) 0.05; };

};

template< typename FiniteDifferenceOperator,
          typename ExactOperator,
          typename Function,
          int MeshSize,
          bool verbose >
bool testFiniteDifferenceOperator()
{
    if( !tnlUnitTestStarter::run<
            tnlPDEOperatorEocTester< 
                FiniteDifferenceOperator,
                ExactOperator,
                Function,
                MeshSize,
                verbose > >() )
        return false;
    return true;
}

template< typename Mesh,
          typename Function,
          typename RealType,
          typename IndexType,
          int XDifference,
          int YDifference,
          int ZDifference,
          int MeshSize,
          bool Verbose >
bool setFiniteDifferenceOperator()
{
    typedef tnlForwardFiniteDifference< Mesh, XDifference, YDifference, ZDifference, RealType, IndexType > ForwardFiniteDifference;
    typedef tnlBackwardFiniteDifference< Mesh, XDifference, YDifference, ZDifference, RealType, IndexType > BackwardFiniteDifference;
    typedef tnlCentralFiniteDifference< Mesh, XDifference, YDifference, ZDifference, RealType, IndexType > CentralFiniteDifference;
    typedef tnlExactDifference< XDifference, YDifference, ZDifference > ExactOperator;
    return ( testFiniteDifferenceOperator< ForwardFiniteDifference, ExactOperator, Function, MeshSize, Verbose >() &&
             testFiniteDifferenceOperator< BackwardFiniteDifference, ExactOperator, Function, MeshSize, Verbose >() &&
             testFiniteDifferenceOperator< CentralFiniteDifference, ExactOperator, Function, MeshSize, Verbose >() );
}

template< typename Mesh,
          typename RealType,
          typename IndexType,
          int XDifference,
          int YDifference,
          int ZDifference,        
          int MeshSize,
          bool Verbose >
bool setFunction()
{
    const int Dimensions = Mesh::meshDimensions;
    typedef tnlExpBumpFunction< Dimensions, RealType >  Function;
    return setFiniteDifferenceOperator< Mesh, Function, RealType, IndexType, XDifference, YDifference, ZDifference, MeshSize, Verbose  >();
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename RealType,
          typename IndexType,
          int MeshSize,
          bool Verbose >
bool setDifferenceOrder()
{
    typedef tnlGrid< 1, MeshReal, Device, MeshIndex > Grid1D;
    typedef tnlGrid< 2, MeshReal, Device, MeshIndex > Grid2D;
    typedef tnlGrid< 3, MeshReal, Device, MeshIndex > Grid3D;
    return ( setFunction< Grid1D, RealType, IndexType, 1, 0, 0, MeshSize, Verbose >() &&
             setFunction< Grid1D, RealType, IndexType, 2, 0, 0, MeshSize, Verbose >() &&
             setFunction< Grid2D, RealType, IndexType, 1, 0, 0, MeshSize, Verbose >() &&
             setFunction< Grid2D, RealType, IndexType, 0, 1, 0, MeshSize, Verbose >() &&
             setFunction< Grid2D, RealType, IndexType, 2, 0, 0, MeshSize, Verbose >() &&            
             setFunction< Grid2D, RealType, IndexType, 0, 2, 0, MeshSize, Verbose >() &&
             setFunction< Grid3D, RealType, IndexType, 1, 0, 0, MeshSize, Verbose >() &&             
             setFunction< Grid3D, RealType, IndexType, 0, 1, 0, MeshSize, Verbose >() &&
             setFunction< Grid3D, RealType, IndexType, 0, 0, 1, MeshSize, Verbose >() &&
             setFunction< Grid3D, RealType, IndexType, 2, 0, 0, MeshSize, Verbose >() &&
             setFunction< Grid3D, RealType, IndexType, 0, 2, 0, MeshSize, Verbose >() &&             
             setFunction< Grid3D, RealType, IndexType, 0, 0, 2, MeshSize, Verbose >() );            
}

bool test()
{
    if( ! setDifferenceOrder< double, tnlHost, int, double, int, 128, true >() )
       return false;
#ifdef HAVE_CUDA
   if( ! setDifferenceOrder< double, tnlCuda, int, double, int, 64, true >() )
      return false;
#endif    
    return true;    
}
