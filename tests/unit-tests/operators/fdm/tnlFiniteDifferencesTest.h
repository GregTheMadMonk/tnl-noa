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

#include "../tnlPDEOperatorEocTest.h"
#include "../tnlPDEOperatorEocUnitTest.h"
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
      static RealType getL1Eoc() 
      {
         if( XDifference < 2 && YDifference < 2 && ZDifference < 2 )
            return ( RealType ) 1.0;
      };
      static RealType getL1Tolerance() { return ( RealType ) 0.05; };

      static RealType getL2Eoc()
      { 
         if( XDifference < 2 && YDifference < 2 && ZDifference < 2 )
            return ( RealType ) 1.0;
      };
      static RealType getL2Tolerance() { return ( RealType ) 0.05; };

      static RealType getMaxEoc()
      {
         if( XDifference < 2 && YDifference < 2 && ZDifference < 2 )
            return ( RealType ) 1.0; 
      };
      static RealType getMaxTolerance() { return ( RealType ) 0.05; };
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
      static RealType getL1Eoc() 
      {
         if( XDifference < 2 && YDifference < 2 && ZDifference < 2 )
            return ( RealType ) 1.0;
      };
      static RealType getL1Tolerance() { return ( RealType ) 0.05; };

      static RealType getL2Eoc()
      { 
         if( XDifference < 2 && YDifference < 2 && ZDifference < 2 )
            return ( RealType ) 1.0;
      };
      static RealType getL2Tolerance() { return ( RealType ) 0.05; };

      static RealType getMaxEoc()
      {
         if( XDifference < 2 && YDifference < 2 && ZDifference < 2 )
            return ( RealType ) 1.0; 
      };
      static RealType getMaxTolerance() { return ( RealType ) 0.05; };

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
      static RealType getL1Eoc() 
      {
         //if( XDifference < 2 && YDifference < 2 && ZDifference < 2 )
            return ( RealType ) 2.0;
      };
      static RealType getL1Tolerance() { return ( RealType ) 0.05; };

      static RealType getL2Eoc()
      { 
         //if( XDifference < 2 && YDifference < 2 && ZDifference < 2 )
            return ( RealType ) 2.0;
      };
      static RealType getL2Tolerance() { return ( RealType ) 0.05; };

      static RealType getMaxEoc()
      {
         //if( XDifference < 2 && YDifference < 2 && ZDifference < 2 )
            return ( RealType ) 2.0; 
      };
      static RealType getMaxTolerance() { return ( RealType ) 0.05; };

};


template< typename ApproximateOperator,
          typename TestFunction,
          bool write = false,
          bool verbose = false >
class tnlFiniteDifferenceTest
   : public tnlPDEOperatorEocTest< ApproximateOperator, TestFunction > 
{
   public:
      
      typedef ApproximateOperator ApproximateOperatorType;
      typedef typename ApproximateOperatorType::ExactOperatorType ExactOperatorType;
      typedef typename ApproximateOperator::MeshType MeshType;
      typedef typename ApproximateOperator::RealType RealType;
      typedef typename ApproximateOperator::IndexType IndexType;
      
      const IndexType coarseMeshSize[ 3 ] = { 1024, 256, 64 };
      
      const RealType  eoc[ 3 ] =       { 2.0,  2.0,  2.0 };
      const RealType  tolerance[ 3 ] = { 0.05, 0.05, 0.05 };      
   
      static tnlString getType()
      { 
         return tnlString( "tnlLinearDiffusionTest< " ) + 
                ApproximateOperator::getType() + ", " +
                TestFunction::getType() + " >";
      }
      
      void setupTest()
      {
         this->setupFunction();
      }
            
      void getApproximationError( const IndexType meshSize,
                                  RealType errors[ 3 ] )
      {
         this->setupMesh( meshSize );
         this->performTest( this->approximateOperator,
                            this->exactOperator,
                            errors,
                            write,
                            verbose );

      }
      
      void runUnitTest()
      {  
         RealType coarseErrors[ 3 ], fineErrors[ 3 ];
         this->getApproximationError( coarseMeshSize[ MeshType::getDimensionsCount() - 1 ], coarseErrors );
         this->getApproximationError( 2 * coarseMeshSize[ MeshType::getDimensionsCount() - 1 ], fineErrors );
         this->checkEoc( coarseErrors, fineErrors, eoc, tolerance, verbose );                            
      }
      
   protected:

      ApproximateOperator approximateOperator;
      
      ExactOperatorType exactOperator;

};


template< typename FiniteDifferenceOperator,
          typename Function,
          bool write,
          bool verbose >
bool testFiniteDifferenceOperator()
{
    typedef tnlFiniteDifferenceTest< FiniteDifferenceOperator, Function, write, verbose > OperatorTest;
#ifdef HAVE_CPPUNIT   
   if( ! tnlUnitTestStarter::run< tnlPDEOperatorEocUnitTest< OperatorTest > >() )
      return false;
   return true;
#endif      
    
}

template< typename Mesh,
          typename Function,
          typename RealType,
          typename IndexType,
          int XDifference,
          int YDifference,
          int ZDifference,
          int MeshSize,
          bool WriteFunctions,
          bool Verbose >
bool setFiniteDifferenceOperator()
{
    typedef tnlForwardFiniteDifference< Mesh, XDifference, YDifference, ZDifference, RealType, IndexType > ForwardFiniteDifference;
    typedef tnlBackwardFiniteDifference< Mesh, XDifference, YDifference, ZDifference, RealType, IndexType > BackwardFiniteDifference;
    typedef tnlCentralFiniteDifference< Mesh, XDifference, YDifference, ZDifference, RealType, IndexType > CentralFiniteDifference;
    
    if( XDifference < 2 && YDifference < 2 && ZDifference < 2 )
      return ( testFiniteDifferenceOperator< ForwardFiniteDifference, Function, WriteFunctions, Verbose >() &&
               testFiniteDifferenceOperator< BackwardFiniteDifference, Function, WriteFunctions, Verbose >() &&
               testFiniteDifferenceOperator< CentralFiniteDifference, Function, WriteFunctions, Verbose >() );
    else
      return ( testFiniteDifferenceOperator< CentralFiniteDifference, Function, WriteFunctions, Verbose >() );
}

template< typename Mesh,
          typename RealType,
          typename IndexType,
          int XDifference,
          int YDifference,
          int ZDifference,        
          int MeshSize,
          bool WriteFunctions,
          bool Verbose >
bool setFunction()
{
    const int Dimensions = Mesh::meshDimensions;
    typedef tnlExpBumpFunction< Dimensions, RealType >  Function;
    return setFiniteDifferenceOperator< Mesh, Function, RealType, IndexType, XDifference, YDifference, ZDifference, MeshSize, WriteFunctions, Verbose  >();
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename RealType,
          typename IndexType,
          int MeshSize,
          bool WriteFunctions,
          bool Verbose >
bool setDifferenceOrder()
{
    typedef tnlGrid< 1, MeshReal, Device, MeshIndex > Grid1D;
    typedef tnlGrid< 2, MeshReal, Device, MeshIndex > Grid2D;
    typedef tnlGrid< 3, MeshReal, Device, MeshIndex > Grid3D;
    return ( setFunction< Grid1D, RealType, IndexType, 1, 0, 0, MeshSize, WriteFunctions, Verbose >() &&
             setFunction< Grid1D, RealType, IndexType, 2, 0, 0, MeshSize, WriteFunctions, Verbose >() &&
             setFunction< Grid2D, RealType, IndexType, 1, 0, 0, MeshSize, WriteFunctions, Verbose >() &&
             setFunction< Grid2D, RealType, IndexType, 0, 1, 0, MeshSize, WriteFunctions, Verbose >() &&
             setFunction< Grid2D, RealType, IndexType, 2, 0, 0, MeshSize, WriteFunctions, Verbose >() &&            
             setFunction< Grid2D, RealType, IndexType, 0, 2, 0, MeshSize, WriteFunctions, Verbose >() &&
             setFunction< Grid3D, RealType, IndexType, 1, 0, 0, MeshSize, WriteFunctions, Verbose >() &&             
             setFunction< Grid3D, RealType, IndexType, 0, 1, 0, MeshSize, WriteFunctions, Verbose >() &&
             setFunction< Grid3D, RealType, IndexType, 0, 0, 1, MeshSize, WriteFunctions, Verbose >() &&
             setFunction< Grid3D, RealType, IndexType, 2, 0, 0, MeshSize, WriteFunctions, Verbose >() &&
             setFunction< Grid3D, RealType, IndexType, 0, 2, 0, MeshSize, WriteFunctions, Verbose >() &&             
             setFunction< Grid3D, RealType, IndexType, 0, 0, 2, MeshSize, WriteFunctions, Verbose >() );            
}

bool test()
{
   const bool writeFunctions( false );
   const bool verbose( false );
   if( ! setDifferenceOrder< double, tnlHost, int, double, int, 64, writeFunctions, verbose >() )
      return false;
#ifdef HAVE_CUDA
   if( ! setDifferenceOrder< double, tnlCuda, int, double, int, 64, writeFunctions, verbose >() )
      return false;
#endif    
    return true;    
}
