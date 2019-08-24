/***************************************************************************
                          DistributedGridTest.cpp  -  description
                             -------------------
    begin                : Sep 6, 2017
    copyright            : (C) 2017 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/


#ifdef HAVE_GTEST  
#include <gtest/gtest.h>

#ifdef HAVE_MPI    

#include <TNL/Communicators/MpiCommunicator.h>
#include <TNL/Functions/MeshFunction.h>
#include <TNL/Meshes/DistributedMeshes/DistributedMesh.h>
#include <TNL/Meshes/DistributedMeshes/SubdomainOverlapsGetter.h>

#include "../../Functions/Functions.h"

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Meshes;
using namespace TNL::Meshes::DistributedMeshes;
using namespace TNL::Functions;
using namespace TNL::Devices;
using namespace TNL::Communicators;


template<typename DofType>
void setDof_1D( DofType &dof, typename DofType::RealType value )
{
   for( int i = 0; i < dof.getSize(); i++ )
      dof[ i ] = value;
}

template<typename DofType>
void check_Boundary_1D(int rank, int nproc, const DofType& dof, typename DofType::RealType expectedValue)
{
    if(rank==0)//Left
    {
        EXPECT_EQ( dof[0], expectedValue) << "Left boundary test failed";
        return;
    }
    
    if(rank==(nproc-1))//Right
    {
        EXPECT_EQ( dof[dof.getSize()-1], expectedValue) << "Right boundary test failed";
        return;
    }
    
};

template<typename DofType>
void check_Overlap_1D(int rank, int nproc, const DofType& dof, typename DofType::RealType expectedValue)
{
    if( rank == 0 )//Left
    {
        EXPECT_EQ( dof[dof.getSize()-1], expectedValue) << "Left boundary node overlap test failed";
        return;
    }
    
    if( rank == ( nproc - 1 ) )
    {
        EXPECT_EQ( dof[0], expectedValue) << "Right boundary node overlap test failed";
        return;
    }
    
    EXPECT_EQ( dof[0], expectedValue) << "left overlap test failed";
    EXPECT_EQ( dof[dof.getSize()-1], expectedValue)<< "right overlap test failed";    
};

template<typename DofType>
void check_Inner_1D(int rank, int nproc, const DofType& dof, typename DofType::RealType expectedValue)
{
    for( int i = 1; i < ( dof.getSize()-2 ); i++ )
        EXPECT_EQ( dof[i], expectedValue) << " " << i;
};

/*
 * Light check of 1D distributed grid and its synchronization. 
 * Number of process is not limited.
 * Overlap is limited to 1
 * Only double is tested as dof Real type -- it may be changed, extend test
 * Global size is hardcoded as 10 -- it can be changed, extend test
 */

typedef MpiCommunicator CommunicatorType;
typedef Grid<1,double,Host,int> GridType;
typedef MeshFunction< GridType > MeshFunctionType;
typedef MeshFunction< GridType, GridType::getMeshDimension(), bool > MaskType;
typedef Vector< double,Host,int> DofType;
typedef Vector< bool, Host, int > MaskDofType;
typedef typename GridType::Cell Cell;
typedef typename GridType::IndexType IndexType; 
typedef typename GridType::PointType PointType; 
typedef DistributedMesh<GridType> DistributedGridType;
using Synchronizer = DistributedMeshSynchronizer< MeshFunctionType >;
     
class DistributedGridTest_1D : public ::testing::Test
{
   protected:

      DistributedMesh< GridType > *distributedGrid;
      DofType dof;
      MaskDofType maskDofs;

      Pointers::SharedPointer< GridType > gridptr;
      Pointers::SharedPointer< MeshFunctionType > meshFunctionPtr;
      Pointers::SharedPointer< MaskType > maskPointer;

      MeshFunctionEvaluator< MeshFunctionType, ConstFunction< double, 1 > > constFunctionEvaluator;
      Pointers::SharedPointer< ConstFunction< double, 1 >, Host > constFunctionPtr;

      MeshFunctionEvaluator< MeshFunctionType, LinearFunction< double, 1 > > linearFunctionEvaluator;
      Pointers::SharedPointer< LinearFunction< double, 1 >, Host > linearFunctionPtr;

      int rank;
      int nproc;

      void SetUp()
      {
         int size=10;
         rank=CommunicatorType::GetRank(CommunicatorType::AllGroup);
         nproc=CommunicatorType::GetSize(CommunicatorType::AllGroup);

         PointType globalOrigin;
         PointType globalProportions;
         GridType globalGrid;

         globalOrigin.x()=-0.5;    
         globalProportions.x()=size;


         globalGrid.setDimensions(size);
         globalGrid.setDomain(globalOrigin,globalProportions);

         typename DistributedGridType::CoordinatesType overlap;
         overlap.setValue(1);
         distributedGrid=new DistributedGridType();

         typename DistributedGridType::SubdomainOverlapsType lowerOverlap, upperOverlap;
         distributedGrid->template setGlobalGrid<CommunicatorType>( globalGrid );
         //distributedGrid->setupGrid(*gridptr);    
         SubdomainOverlapsGetter< GridType, CommunicatorType >::
            getOverlaps( distributedGrid, lowerOverlap, upperOverlap, 1 );
         distributedGrid->setOverlaps( lowerOverlap, upperOverlap );

         distributedGrid->setupGrid(*gridptr);
         dof.setSize( gridptr->template getEntitiesCount< Cell >() );

         meshFunctionPtr->bind(gridptr,dof);

         constFunctionPtr->Number=rank;
      }
      
      void SetUpPeriodicBoundaries()
      {
         typename DistributedGridType::SubdomainOverlapsType lowerOverlap, upperOverlap;
         SubdomainOverlapsGetter< GridType, CommunicatorType >::
            getOverlaps( distributedGrid, lowerOverlap, upperOverlap, 1 );
         distributedGrid->setOverlaps( lowerOverlap, upperOverlap );
         distributedGrid->setupGrid(*gridptr);         
      }

      void TearDown()
      {
         delete distributedGrid;
      }
};

TEST_F( DistributedGridTest_1D, isBoundaryDomainTest )
{
   if( rank == 0 || rank == nproc - 1 )
      EXPECT_TRUE( distributedGrid->isBoundarySubdomain() );
   else
      EXPECT_FALSE( distributedGrid->isBoundarySubdomain() );
}

TEST_F(DistributedGridTest_1D, evaluateAllEntities)
{
   //Check Traversars
   //All entities, witout overlap
   setDof_1D( dof,-1);
   constFunctionEvaluator.evaluateAllEntities( meshFunctionPtr , constFunctionPtr );
   //Printer<GridType,DofType>::print_dof(rank,*gridptr,dof);
   check_Boundary_1D(rank, nproc, dof, rank);
   check_Overlap_1D(rank, nproc, dof, -1);
   check_Inner_1D(rank, nproc, dof, rank);
}

TEST_F(DistributedGridTest_1D, evaluateBoundaryEntities)
{
   //Boundary entities, witout overlap
   setDof_1D(dof,-1);
   constFunctionEvaluator.evaluateBoundaryEntities( meshFunctionPtr , constFunctionPtr );
   check_Boundary_1D(rank, nproc, dof, rank);
   check_Overlap_1D(rank, nproc, dof, -1);
   check_Inner_1D(rank, nproc, dof, -1);
}

TEST_F(DistributedGridTest_1D, evaluateInteriorEntities)
{
   //Inner entities, witout overlap
   setDof_1D(dof,-1);
   constFunctionEvaluator.evaluateInteriorEntities( meshFunctionPtr , constFunctionPtr );
   check_Boundary_1D(rank, nproc, dof, -1);
   check_Overlap_1D(rank, nproc, dof, -1);
   check_Inner_1D(rank, nproc, dof, rank);
}    

TEST_F(DistributedGridTest_1D, SynchronizerNeighborsTest )
{
   setDof_1D(dof,-1);
   constFunctionEvaluator.evaluateAllEntities( meshFunctionPtr , constFunctionPtr );
   meshFunctionPtr->template synchronize<CommunicatorType>();

   if(rank!=0) {
      EXPECT_EQ((dof)[0],rank-1)<< "Left Overlap was filled by wrong process.";
   }
   if(rank!=nproc-1) {
      EXPECT_EQ((dof)[dof.getSize()-1],rank+1)<< "Right Overlap was filled by wrong process.";
   }
}

TEST_F(DistributedGridTest_1D, EvaluateLinearFunction )
{
   //fill mesh function with linear function (physical center of cell corresponds with its coordinates in grid) 
   setDof_1D(dof,-1);
   linearFunctionEvaluator.evaluateAllEntities(meshFunctionPtr, linearFunctionPtr);
   meshFunctionPtr->template synchronize<CommunicatorType>();

   auto entity = gridptr->template getEntity< Cell >(0);
   entity.refresh();
   EXPECT_EQ(meshFunctionPtr->getValue(entity), (*linearFunctionPtr)(entity)) << "Linear function Overlap error on left Edge.";

   auto entity2= gridptr->template getEntity< Cell >((dof).getSize()-1);
   entity2.refresh();
   EXPECT_EQ(meshFunctionPtr->getValue(entity), (*linearFunctionPtr)(entity)) << "Linear function Overlap error on right Edge.";
}

TEST_F(DistributedGridTest_1D, SynchronizePeriodicNeighborsWithoutMask )
{
   // Setup periodic boundaries
   // TODO: I do not know how to do it better with GTEST
   typename DistributedGridType::SubdomainOverlapsType lowerOverlap, upperOverlap;
   SubdomainOverlapsGetter< GridType, CommunicatorType >::
      getOverlaps( distributedGrid, lowerOverlap, upperOverlap, 1, 1, 1 );
   distributedGrid->setOverlaps( lowerOverlap, upperOverlap );
   distributedGrid->setupGrid(*gridptr);
   dof.setSize( gridptr->template getEntitiesCount< Cell >() );
   maskDofs.setSize( gridptr->template getEntitiesCount< Cell >() );
   meshFunctionPtr->bind( gridptr, dof );
   maskPointer->bind( gridptr, maskDofs );
   
   setDof_1D( dof, -rank-1 );
   maskDofs.setValue( true );
   //meshFunctionPtr->getSynchronizer().setPeriodicBoundariesCopyDirection( Synchronizer::OverlapToBoundary );
   meshFunctionPtr->template synchronize<CommunicatorType>( true );

   if( rank == 0 ) {
      EXPECT_EQ( dof[ 0 ], -nproc ) << "Left Overlap was filled by wrong process.";
   }
   if( rank == nproc-1 ) {
      EXPECT_EQ( dof[ dof.getSize() - 1 ], -1 )<< "Right Overlap was filled by wrong process.";
   }
}

TEST_F(DistributedGridTest_1D, SynchronizePeriodicNeighborsWithActiveMask )
{
   // Setup periodic boundaries
   // TODO: I do not know how to do it better with GTEST
   typename DistributedGridType::SubdomainOverlapsType lowerOverlap, upperOverlap;
   SubdomainOverlapsGetter< GridType, CommunicatorType >::
      getOverlaps( distributedGrid, lowerOverlap, upperOverlap, 1, 1, 1 );
   distributedGrid->setOverlaps( lowerOverlap, upperOverlap );
   distributedGrid->setupGrid(*gridptr);
   dof.setSize( gridptr->template getEntitiesCount< Cell >() );
   maskDofs.setSize( gridptr->template getEntitiesCount< Cell >() );
   meshFunctionPtr->bind( gridptr, dof );
   maskPointer->bind( gridptr, maskDofs );
   
   setDof_1D( dof, -rank-1 );
   maskDofs.setValue( true );
   //constFunctionEvaluator.evaluateAllEntities( meshFunctionPtr, constFunctionPtr );
   //meshFunctionPtr->getSynchronizer().setPeriodicBoundariesCopyDirection( Synchronizer::OverlapToBoundary );
   meshFunctionPtr->template synchronize<CommunicatorType>( true, maskPointer );
   if( rank == 0 ) {
      EXPECT_EQ( dof[ 0 ], -nproc ) << "Left Overlap was filled by wrong process.";
   }
   if( rank == nproc-1 ) {
      EXPECT_EQ( dof[ dof.getSize() - 1 ], -1 )<< "Right Overlap was filled by wrong process.";
   }
}

// TODO: Fix tests with overlap-to-boundary direction and masks
/*
TEST_F(DistributedGridTest_1D, SynchronizePeriodicNeighborsWithInactiveMaskOnLeft )
{
   // Setup periodic boundaries
   // TODO: I do not know how to do it better with GTEST
   typename DistributedGridType::SubdomainOverlapsType lowerOverlap, upperOverlap;
   SubdomainOverlapsGetter< GridType, CommunicatorType >::
      getOverlaps( distributedGrid, lowerOverlap, upperOverlap, 1, 1, 1 );
   distributedGrid->setOverlaps( lowerOverlap, upperOverlap );
   distributedGrid->setupGrid(*gridptr);
   dof.setSize( gridptr->template getEntitiesCount< Cell >() );
   maskDofs.setSize( gridptr->template getEntitiesCount< Cell >() );
   meshFunctionPtr->bind( gridptr, dof );
   maskPointer->bind( gridptr, maskDofs );

   setDof_1D( dof, -rank-1 );
   maskDofs.setValue( true );
   maskDofs.setElement( 1, false );
   //constFunctionEvaluator.evaluateAllEntities( meshFunctionPtr , constFunctionPtr );
   //meshFunctionPtr->getSynchronizer().setPeriodicBoundariesCopyDirection( Synchronizer::OverlapToBoundary );
   TNL_MPI_PRINT( "#### " << dof );
   meshFunctionPtr->template synchronize<CommunicatorType>( true, maskPointer );
   TNL_MPI_PRINT( ">>> " << dof );
   
   if( rank == 0 )
      EXPECT_EQ( dof[ 0 ], 0 ) << "Left Overlap was filled by wrong process.";
   if( rank == nproc-1 )
      EXPECT_EQ( dof[ dof.getSize() - 1 ], -1 )<< "Right Overlap was filled by wrong process.";
}

TEST_F(DistributedGridTest_1D, SynchronizePeriodicNeighborsWithInactiveMask )
{
   // Setup periodic boundaries
   // TODO: I do not know how to do it better with GTEST
   typename DistributedGridType::SubdomainOverlapsType lowerOverlap, upperOverlap;
   SubdomainOverlapsGetter< GridType, CommunicatorType >::
      getOverlaps( distributedGrid, lowerOverlap, upperOverlap, 1, 1, 1 );
   distributedGrid->setOverlaps( lowerOverlap, upperOverlap );
   distributedGrid->setupGrid(*gridptr);
   dof.setSize( gridptr->template getEntitiesCount< Cell >() );
   maskDofs.setSize( gridptr->template getEntitiesCount< Cell >() );
   meshFunctionPtr->bind( gridptr, dof );
   maskPointer->bind( gridptr, maskDofs );

   setDof_1D( dof, -rank-1 );
   maskDofs.setValue( true );
   maskDofs.setElement( 1, false );   
   maskDofs.setElement( dof.getSize() - 2, false );
   //constFunctionEvaluator.evaluateAllEntities( meshFunctionPtr , constFunctionPtr );
   //meshFunctionPtr->getSynchronizer().setPeriodicBoundariesCopyDirection( Synchronizer::OverlapToBoundary );
   meshFunctionPtr->template synchronize<CommunicatorType>( true, maskPointer );
   
   if( rank == 0 )
      EXPECT_EQ( dof[ 0 ], 0 ) << "Left Overlap was filled by wrong process.";
   if( rank == nproc-1 )
      EXPECT_EQ( dof[ dof.getSize() - 1 ], nproc - 1 )<< "Right Overlap was filled by wrong process.";   
   
}
*/

TEST_F(DistributedGridTest_1D, SynchronizePeriodicBoundariesLinearTest )
{
   // Setup periodic boundaries
   // TODO: I do not know how to do it better with GTEST - additional setup 
   // of the periodic boundaries
   typename DistributedGridType::SubdomainOverlapsType lowerOverlap, upperOverlap;
   SubdomainOverlapsGetter< GridType, CommunicatorType >::
      getOverlaps( distributedGrid, lowerOverlap, upperOverlap, 1, 1, 1 );
   distributedGrid->setOverlaps( lowerOverlap, upperOverlap );
   distributedGrid->setupGrid(*gridptr);
   dof.setSize( gridptr->template getEntitiesCount< Cell >() );
   meshFunctionPtr->bind( gridptr, dof );

   setDof_1D(dof, -rank-1 );
   linearFunctionEvaluator.evaluateAllEntities( meshFunctionPtr , linearFunctionPtr );

   meshFunctionPtr->template synchronize<CommunicatorType>( true );

   auto entity = gridptr->template getEntity< Cell >( 0 );
   auto entity2= gridptr->template getEntity< Cell >( (dof).getSize() - 1 );
   entity.refresh();
   entity2.refresh();
   
   if( rank == 0 ) {
      EXPECT_EQ( meshFunctionPtr->getValue(entity), 9 ) << "Linear function Overlap error on left Edge.";
   }
   if( rank == nproc - 1 ) {
      EXPECT_EQ( meshFunctionPtr->getValue(entity2), 0 ) << "Linear function Overlap error on right Edge.";
   }
}
#endif

#endif

#include "../../main_mpi.h"
