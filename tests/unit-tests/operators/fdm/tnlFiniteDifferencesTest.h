/***************************************************************************
                          FiniteDifferencesTest.h  -  description
                             -------------------
    begin                : Jan 12, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/tnlConfig.h>
#include <TNL/Devices/Host.h>
#include <cstdlib>

#include "../tnlPDEOperatorEocTest.h"
#include "../tnlPDEOperatorEocUnitTest.h"
#include "../../tnlUnitTestStarter.h"
#include <TNL/Meshes/Grid.h>
#include <TNL/Operators/fdm/BackwardFiniteDifference.h>
#include <TNL/Operators/fdm/ForwardFiniteDifference.h>
#include <TNL/Operators/fdm/CentralFiniteDifference.h>
#include <TNL/Operators/fdm/ExactDifference.h>
#include "../tnlPDEOperatorEocTestResult.h"
#include <TNL/Functions/Analytic/ExpBump.h>

using namespace TNL;

template< typename ApproximateOperator >
class tnlFinitDifferenceEocTestResults
{
   public:
 
      typedef typename ApproximateOperator::RealType RealType;
 
      const RealType  eoc[ 3 ] =       { 1.0,  1.0,  1.0 };
      const RealType  tolerance[ 3 ] = { 0.05, 0.05, 0.05 };

};

template< typename MeshType,
          typename RealType,
          int XDifference,
          int YDifference,
          int ZDifference,
          typename IndexType >
class tnlFinitDifferenceEocTestResults< Operators::CentralFiniteDifference< MeshType, XDifference, YDifference, ZDifference, RealType, IndexType > >
{
   public:
 
      const RealType  eoc[ 3 ] =       { 2.0,  2.0,  2.0 };
      const RealType  tolerance[ 3 ] = { 0.05, 0.05, 0.05 };
};

template< typename ApproximateOperator,
          typename TestFunction,
          bool write = false,
          bool verbose = false >
class tnlFiniteDifferenceTest
   : public tnlPDEOperatorEocTest< ApproximateOperator, TestFunction >,
     public tnlFinitDifferenceEocTestResults< ApproximateOperator >
{
   public:
 
      typedef ApproximateOperator ApproximateOperatorType;
      typedef typename ApproximateOperatorType::ExactOperatorType ExactOperatorType;
      typedef typename ApproximateOperator::MeshType MeshType;
      typedef typename ApproximateOperator::RealType RealType;
      typedef typename ApproximateOperator::IndexType IndexType;
 
      const IndexType coarseMeshSize[ 3 ] = { 1024, 256, 64 };
 
 
      static String getType()
      {
         return String( "LinearDiffusionTest< " ) +
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
         this->getApproximationError( coarseMeshSize[ MeshType::getDimension() - 1 ], coarseErrors );
         this->getApproximationError( 2 * coarseMeshSize[ MeshType::getDimension() - 1 ], fineErrors );
         this->checkEoc( coarseErrors, fineErrors, this->eoc, this->tolerance, verbose );
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
    typedef Operators::ForwardFiniteDifference< Mesh, XDifference, YDifference, ZDifference, RealType, IndexType > ForwardFiniteDifference;
    typedef Operators::BackwardFiniteDifference< Mesh, XDifference, YDifference, ZDifference, RealType, IndexType > BackwardFiniteDifference;
    typedef Operators::CentralFiniteDifference< Mesh, XDifference, YDifference, ZDifference, RealType, IndexType > CentralFiniteDifference;
 
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
    const int Dimension = Mesh::meshDimension;
    typedef Functions::Analytic::ExpBump< Dimension, RealType >  Function;
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
    typedef Meshes::Grid< 1, MeshReal, Device, MeshIndex > Grid1D;
    typedef Meshes::Grid< 2, MeshReal, Device, MeshIndex > Grid2D;
    typedef Meshes::Grid< 3, MeshReal, Device, MeshIndex > Grid3D;
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
   const bool verbose( true );
   if( ! setDifferenceOrder< double, Devices::Host, int, double, int, 64, writeFunctions, verbose >() )
      return false;
#ifdef HAVE_CUDA
   if( ! setDifferenceOrder< double, Devices::Cuda, int, double, int, 64, writeFunctions, verbose >() )
      return false;
#endif
    return true;
}

int main( int argc, char* argv[] )
{
#ifdef HAVE_CPPUNIT
    return test();
#else
   return EXIT_FAILURE;
#endif
}
