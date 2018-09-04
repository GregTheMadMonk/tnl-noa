#ifdef HAVE_GTEST  
#include <gtest/gtest.h>

#ifdef HAVE_MPI  

#include <TNL/Devices/Host.h> 
#include <TNL/Functions/CutMeshFunction.h>
#include <TNL/Communicators/MpiCommunicator.h>
#include <TNL/Meshes/DistributedMeshes/DistributedGridIO.h>
#include <TNL/Meshes/DistributedMeshes/SubdomainOverlapsGetter.h>

#include "../../Functions/Functions.h"

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Meshes;
using namespace TNL::Meshes::DistributedMeshes;
using namespace TNL::Devices;
using namespace TNL::Communicators;

typedef MpiCommunicator CommunicatorType;

//======================================DATA===============================================================
TEST(CutDistributedMeshFunction, 2D_Data)
{
   typedef Grid<2, double,Host,int> MeshType;
   typedef Grid<1, double,Host,int> CutMeshType;
   typedef DistributedMesh<MeshType> DistributedMeshType;
   typedef DistributedMesh<CutMeshType> CutDistributedMeshType;


   typedef Vector<double,Host,int> DofType;

   typedef typename MeshType::PointType PointType;
   typedef typename MeshType::Cell Cell;

   typedef LinearFunction<double,2> LinearFunctionType;

   MeshType globalOriginalGrid; 
   PointType origin;
   origin.setValue(-0.5);
   PointType proportions;
   proportions.setValue(10);
   globalOriginalGrid.setDimensions(proportions);
   globalOriginalGrid.setDomain(origin,proportions);

   typename DistributedMeshType::CoordinatesType overlap;
   
   DistributedMeshType distributedGrid;
   distributedGrid.setDomainDecomposition( typename DistributedMeshType::CoordinatesType( 3, 4 ) );
   distributedGrid.template setGlobalGrid<CommunicatorType>(globalOriginalGrid);
   typename DistributedMeshType::SubdomainOverlapsType lowerOverlap, upperOverlap;
   SubdomainOverlapsGetter< MeshType, CommunicatorType >::getOverlaps( &distributedGrid, lowerOverlap, upperOverlap, 1 );
   distributedGrid.setOverlaps( lowerOverlap, upperOverlap );


   SharedPointer<MeshType> originalGrid;
   distributedGrid.setupGrid(*originalGrid);

   DofType dof(originalGrid->template getEntitiesCount< Cell >());
   dof.setValue(0); 

   SharedPointer<MeshFunction<MeshType>> meshFunctionptr;
   meshFunctionptr->bind(originalGrid,dof);

   MeshFunctionEvaluator< MeshFunction<MeshType>, LinearFunctionType > linearFunctionEvaluator;
   SharedPointer< LinearFunctionType, Host > linearFunctionPtr;
   linearFunctionEvaluator.evaluateAllEntities(meshFunctionptr , linearFunctionPtr);

   meshFunctionptr->template synchronize<CommunicatorType>();
 
   //Prepare Mesh Function parts for Cut 
   CutDistributedMeshType cutDistributedGrid;
   SharedPointer<CutMeshType> cutGrid;
   cutGrid->setDistMesh(&cutDistributedGrid);
   DofType cutDof(0);
   bool inCut=CutMeshFunction<CommunicatorType, MeshFunction<MeshType>,CutMeshType,DofType>::Cut(
            *meshFunctionptr,*cutGrid, cutDof, 
            StaticVector<1,int>(1),
            StaticVector<1,int>(0),
            StaticVector<1,typename CutMeshType::IndexType>(5) );

   if(inCut)
   {
       MeshFunction<CutMeshType> cutMeshFunction;
            cutMeshFunction.bind(cutGrid,cutDof); 

        for(int i=0;i<originalGrid->getDimensions().y();i++)
        {
               typename MeshType::Cell fromEntity(meshFunctionptr->getMesh());
               typename CutMeshType::Cell outEntity(*cutGrid);
               
                fromEntity.getCoordinates().x()=5-distributedGrid.getGlobalBegin().x();
                fromEntity.getCoordinates().y()=i;
                outEntity.getCoordinates().x()=i;

                fromEntity.refresh();
                outEntity.refresh();

                EXPECT_EQ(cutDof[outEntity.getIndex()],dof[fromEntity.getIndex()]) <<i  <<" Chyba";
        }

    }
}

TEST(CutDistributedMeshFunction, 3D_1_Data)
{
   typedef Grid<3, double,Host,int> MeshType;
   typedef Grid<1, double,Host,int> CutMeshType;
   typedef DistributedMesh<MeshType> DistributedMeshType;
   typedef DistributedMesh<CutMeshType> CutDistributedMeshType;


   typedef Vector<double,Host,int> DofType;

   typedef typename MeshType::PointType PointType;
   typedef typename MeshType::Cell Cell;

   typedef LinearFunction<double,3> LinearFunctionType;

   MeshType globalOriginalGrid; 
   PointType origin;
   origin.setValue(-0.5);
   PointType proportions;
   proportions.setValue(10);
   globalOriginalGrid.setDimensions(proportions);
   globalOriginalGrid.setDomain(origin,proportions);

   DistributedMeshType distributedGrid;
   distributedGrid.setDomainDecomposition( typename DistributedMeshType::CoordinatesType( 2, 2, 3 ) );
   distributedGrid.template setGlobalGrid<CommunicatorType>(globalOriginalGrid);
   typename DistributedMeshType::SubdomainOverlapsType lowerOverlap, upperOverlap;
   SubdomainOverlapsGetter< MeshType, CommunicatorType >::getOverlaps( &distributedGrid, lowerOverlap, upperOverlap, 1 );
   distributedGrid.setOverlaps( lowerOverlap, upperOverlap );

   SharedPointer<MeshType> originalGrid;
   distributedGrid.setupGrid(*originalGrid);

   DofType dof(originalGrid->template getEntitiesCount< Cell >());
   dof.setValue(0); 

   SharedPointer<MeshFunction<MeshType>> meshFunctionptr;
   meshFunctionptr->bind(originalGrid,dof);

   MeshFunctionEvaluator< MeshFunction<MeshType>, LinearFunctionType > linearFunctionEvaluator;
   SharedPointer< LinearFunctionType, Host > linearFunctionPtr;
   linearFunctionEvaluator.evaluateAllEntities(meshFunctionptr , linearFunctionPtr);

   meshFunctionptr->template synchronize<CommunicatorType>();

   //Prepare Mesh Function parts for Cut 
   CutDistributedMeshType cutDistributedGrid;
   SharedPointer<CutMeshType> cutGrid;
   cutGrid->setDistMesh(&cutDistributedGrid);
   DofType cutDof(0);
   bool inCut=CutMeshFunction<CommunicatorType, MeshFunction<MeshType>,CutMeshType,DofType>::Cut(
            *meshFunctionptr,*cutGrid, cutDof, 
            StaticVector<1,int>(2),
            StaticVector<2,int>(1,0),
            StaticVector<2,typename CutMeshType::IndexType>(3,4) );

   if(inCut)
   {
       MeshFunction<CutMeshType> cutMeshFunction;
            cutMeshFunction.bind(cutGrid,cutDof); 

        for(int i=0;i<originalGrid->getDimensions().z();i++)
        {
               typename MeshType::Cell fromEntity(meshFunctionptr->getMesh());
               typename CutMeshType::Cell outEntity(*cutGrid);
               
                fromEntity.getCoordinates().x()=4-distributedGrid.getGlobalBegin().x();
                fromEntity.getCoordinates().y()=3-distributedGrid.getGlobalBegin().y();
                fromEntity.getCoordinates().z()=i;
                outEntity.getCoordinates().x()=i;

                fromEntity.refresh();
                outEntity.refresh();

                EXPECT_EQ(cutDof[outEntity.getIndex()],dof[fromEntity.getIndex()]) <<i  <<" Chyba";
        }
    }
}

TEST(CutDistributedMeshFunction, 3D_2_Data)
{
   typedef Grid<3, double,Host,int> MeshType;
   typedef Grid<2, double,Host,int> CutMeshType;
   typedef DistributedMesh<MeshType> DistributedMeshType;
   typedef DistributedMesh<CutMeshType> CutDistributedMeshType;


   typedef Vector<double,Host,int> DofType;

   typedef typename MeshType::PointType PointType;
   typedef typename MeshType::Cell Cell;

   typedef LinearFunction<double,3> LinearFunctionType;

   MeshType globalOriginalGrid; 
   PointType origin;
   origin.setValue(-0.5);
   PointType proportions;
   proportions.setValue(10);
   globalOriginalGrid.setDimensions(proportions);
   globalOriginalGrid.setDomain(origin,proportions);

   DistributedMeshType distributedGrid;
   distributedGrid.setDomainDecomposition( typename DistributedMeshType::CoordinatesType( 2, 2, 3 ) );
   distributedGrid.template setGlobalGrid<CommunicatorType>(globalOriginalGrid);
   typename DistributedMeshType::SubdomainOverlapsType lowerOverlap, upperOverlap;
   SubdomainOverlapsGetter< MeshType, CommunicatorType >::getOverlaps( &distributedGrid, lowerOverlap, upperOverlap, 1 );
   distributedGrid.setOverlaps( lowerOverlap, upperOverlap );

   SharedPointer<MeshType> originalGrid;
   distributedGrid.setupGrid(*originalGrid);

   DofType dof(originalGrid->template getEntitiesCount< Cell >());
   dof.setValue(0); 

   SharedPointer<MeshFunction<MeshType>> meshFunctionptr;
   meshFunctionptr->bind(originalGrid,dof);

   MeshFunctionEvaluator< MeshFunction<MeshType>, LinearFunctionType > linearFunctionEvaluator;
   SharedPointer< LinearFunctionType, Host > linearFunctionPtr;
   linearFunctionEvaluator.evaluateAllEntities(meshFunctionptr , linearFunctionPtr);

   meshFunctionptr->template synchronize<CommunicatorType>();

   //Prepare Mesh Function parts for Cut 
   CutDistributedMeshType cutDistributedGrid;
   SharedPointer<CutMeshType> cutGrid;
   cutGrid->setDistMesh(&cutDistributedGrid);
   DofType cutDof(0);
   bool inCut=CutMeshFunction<CommunicatorType, MeshFunction<MeshType>,CutMeshType,DofType>::Cut(
            *meshFunctionptr,*cutGrid, cutDof, 
            StaticVector<2,int>(0,2),
            StaticVector<1,int>(1),
            StaticVector<1,typename CutMeshType::IndexType>(4) );

   if(inCut)
   {
       MeshFunction<CutMeshType> cutMeshFunction;
            cutMeshFunction.bind(cutGrid,cutDof); 

        for(int i=0;i<originalGrid->getDimensions().z();i++)
        {
            for(int j=0;j<originalGrid->getDimensions().x();j++)
            {
               typename MeshType::Cell fromEntity(meshFunctionptr->getMesh());
               typename CutMeshType::Cell outEntity(*cutGrid);
               
                fromEntity.getCoordinates().x()=j;
                fromEntity.getCoordinates().y()=4-distributedGrid.getGlobalBegin().y();
                fromEntity.getCoordinates().z()=i;

                outEntity.getCoordinates().x()=j;
                outEntity.getCoordinates().y()=i;

                fromEntity.refresh();
                outEntity.refresh();

                EXPECT_EQ(cutDof[outEntity.getIndex()],dof[fromEntity.getIndex()]) <<i  <<" Chyba";
            }
        }
    }
}

//================================Synchronization========================================================
TEST(CutDistributedMeshFunction, 2D_Synchronization)
{
   typedef Grid<2, double,Host,int> MeshType;
   typedef Grid<1, double,Host,int> CutMeshType;
   typedef DistributedMesh<MeshType> DistributedMeshType;
   typedef DistributedMesh<CutMeshType> CutDistributedMeshType;


   typedef Vector<double,Host,int> DofType;

   typedef typename MeshType::PointType PointType;
   typedef typename MeshType::Cell Cell;

   typedef LinearFunction<double,2> LinearFunctionType;

   MeshType globalOriginalGrid; 
   PointType origin;
   origin.setValue(-0.5);
   PointType proportions;
   proportions.setValue(10);
   globalOriginalGrid.setDimensions(proportions);
   globalOriginalGrid.setDomain(origin,proportions);

   DistributedMeshType distributedGrid;
   distributedGrid.setDomainDecomposition( typename DistributedMeshType::CoordinatesType( 3, 4 ) );
   distributedGrid.template setGlobalGrid<CommunicatorType>(globalOriginalGrid);
   typename DistributedMeshType::SubdomainOverlapsType lowerOverlap, upperOverlap;
   SubdomainOverlapsGetter< MeshType, CommunicatorType >::getOverlaps( &distributedGrid, lowerOverlap, upperOverlap, 1 );
   distributedGrid.setOverlaps( lowerOverlap, upperOverlap );

   SharedPointer<MeshType> originalGrid;
   distributedGrid.setupGrid(*originalGrid);

   DofType dof(originalGrid->template getEntitiesCount< Cell >());
   dof.setValue(0); 

   SharedPointer<MeshFunction<MeshType>> meshFunctionptr;
   meshFunctionptr->bind(originalGrid,dof);

   MeshFunctionEvaluator< MeshFunction<MeshType>, LinearFunctionType > linearFunctionEvaluator;
   SharedPointer< LinearFunctionType, Host > linearFunctionPtr;
   linearFunctionEvaluator.evaluateAllEntities(meshFunctionptr , linearFunctionPtr);

   //Prepare Mesh Function parts for Cut 
   CutDistributedMeshType cutDistributedGrid;
   SharedPointer<CutMeshType> cutGrid;
   cutGrid->setDistMesh(&cutDistributedGrid);
   DofType cutDof(0);
   bool inCut=CutMeshFunction<CommunicatorType, MeshFunction<MeshType>,CutMeshType,DofType>::Cut(
            *meshFunctionptr,*cutGrid, cutDof, 
            StaticVector<1,int>(1),
            StaticVector<1,int>(0),
            StaticVector<1,typename CutMeshType::IndexType>(5) );

   if(inCut)
   {
       MeshFunction<CutMeshType> cutMeshFunction;
            cutMeshFunction.bind(cutGrid,cutDof); 

        cutMeshFunction.template synchronize<CommunicatorType>();

        typename MeshType::Cell fromEntity(meshFunctionptr->getMesh());
        typename CutMeshType::Cell outEntity(*cutGrid);
                
        fromEntity.getCoordinates().x()=5-distributedGrid.getGlobalBegin().x();
        fromEntity.getCoordinates().y()=0;
        outEntity.getCoordinates().x()=0;
        fromEntity.refresh();
        outEntity.refresh();

        EXPECT_EQ(cutMeshFunction.getValue(outEntity), (*linearFunctionPtr)(fromEntity)) << "Error in Left overlap";

        fromEntity.getCoordinates().x()=5-distributedGrid.getGlobalBegin().x();
        fromEntity.getCoordinates().y()=(cutDof).getSize()-1;
        outEntity.getCoordinates().x()=(cutDof).getSize()-1;
        fromEntity.refresh();
        outEntity.refresh();

        EXPECT_EQ(cutMeshFunction.getValue(outEntity), (*linearFunctionPtr)(fromEntity)) << "Error in Right overlap";

    }
}

TEST(CutDistributedMeshFunction, 3D_1_Synchronization)
{
   typedef Grid<3, double,Host,int> MeshType;
   typedef Grid<1, double,Host,int> CutMeshType;
   typedef DistributedMesh<MeshType> DistributedMeshType;
   typedef DistributedMesh<CutMeshType> CutDistributedMeshType;


   typedef Vector<double,Host,int> DofType;

   typedef typename MeshType::PointType PointType;
   typedef typename MeshType::Cell Cell;

   typedef LinearFunction<double,3> LinearFunctionType;

   MeshType globalOriginalGrid; 
   PointType origin;
   origin.setValue(-0.5);
   PointType proportions;
   proportions.setValue(10);
   globalOriginalGrid.setDimensions(proportions);
   globalOriginalGrid.setDomain(origin,proportions);

   DistributedMeshType distributedGrid;
   distributedGrid.setDomainDecomposition( typename DistributedMeshType::CoordinatesType( 2,2,3 ) );
   distributedGrid.template setGlobalGrid<CommunicatorType>( globalOriginalGrid );
   typename DistributedMeshType::SubdomainOverlapsType lowerOverlap, upperOverlap;
   SubdomainOverlapsGetter< MeshType, CommunicatorType >::getOverlaps( &distributedGrid, lowerOverlap, upperOverlap, 1 );
   distributedGrid.setOverlaps( lowerOverlap, upperOverlap );

   SharedPointer<MeshType> originalGrid;
   distributedGrid.setupGrid(*originalGrid);

   DofType dof(originalGrid->template getEntitiesCount< Cell >());
   dof.setValue(0); 

   SharedPointer<MeshFunction<MeshType>> meshFunctionptr;
   meshFunctionptr->bind(originalGrid,dof);

   MeshFunctionEvaluator< MeshFunction<MeshType>, LinearFunctionType > linearFunctionEvaluator;
   SharedPointer< LinearFunctionType, Host > linearFunctionPtr;
   linearFunctionEvaluator.evaluateAllEntities(meshFunctionptr , linearFunctionPtr);

   //Prepare Mesh Function parts for Cut 
   CutDistributedMeshType cutDistributedGrid;
   SharedPointer<CutMeshType> cutGrid;
   cutGrid->setDistMesh(&cutDistributedGrid);
   DofType cutDof(0);
   bool inCut=CutMeshFunction<CommunicatorType, MeshFunction<MeshType>,CutMeshType,DofType>::Cut(
            *meshFunctionptr,*cutGrid, cutDof, 
            StaticVector<1,int>(1),
            StaticVector<2,int>(0,2),
            StaticVector<2,typename CutMeshType::IndexType>(4,4) );

   if(inCut)
   {
       MeshFunction<CutMeshType> cutMeshFunction;
            cutMeshFunction.bind(cutGrid,cutDof); 

        cutMeshFunction.template synchronize<CommunicatorType>();

        typename MeshType::Cell fromEntity(meshFunctionptr->getMesh());
        typename CutMeshType::Cell outEntity(*cutGrid);
                
        fromEntity.getCoordinates().x()=4-distributedGrid.getGlobalBegin().x();
        fromEntity.getCoordinates().z()=4-distributedGrid.getGlobalBegin().x();
        fromEntity.getCoordinates().y()=0;
        outEntity.getCoordinates().x()=0;
        fromEntity.refresh();
        outEntity.refresh();

        EXPECT_EQ(cutMeshFunction.getValue(outEntity), (*linearFunctionPtr)(fromEntity)) << "Error in Left overlap";

        fromEntity.getCoordinates().x()=4-distributedGrid.getGlobalBegin().x();
        fromEntity.getCoordinates().z()=4-distributedGrid.getGlobalBegin().x();
        fromEntity.getCoordinates().y()=(cutDof).getSize()-1;
        outEntity.getCoordinates().x()=(cutDof).getSize()-1;
        fromEntity.refresh();
        outEntity.refresh();

        EXPECT_EQ(cutMeshFunction.getValue(outEntity), (*linearFunctionPtr)(fromEntity)) << "Error in Right overlap";

    }
}

TEST(CutDistributedMeshFunction, 3D_2_Synchronization)
{
   typedef Grid<3, double,Host,int> MeshType;
   typedef Grid<2, double,Host,int> CutMeshType;
   typedef DistributedMesh<MeshType> DistributedMeshType;
   typedef DistributedMesh<CutMeshType> CutDistributedMeshType;


   typedef Vector<double,Host,int> DofType;

   typedef typename MeshType::PointType PointType;
   typedef typename MeshType::Cell Cell;

   typedef LinearFunction<double,3> LinearFunctionType;

   MeshType globalOriginalGrid; 
   PointType origin;
   origin.setValue(-0.5);
   PointType proportions;
   proportions.setValue(10);
   globalOriginalGrid.setDimensions(proportions);
   globalOriginalGrid.setDomain(origin,proportions);

   typename DistributedMeshType::CoordinatesType overlap;
   overlap.setValue(1);
   DistributedMeshType distributedGrid;
   distributedGrid.setDomainDecomposition( typename DistributedMeshType::CoordinatesType( 2,2,3 ) );
   distributedGrid.template setGlobalGrid<CommunicatorType>(globalOriginalGrid);
   typename DistributedMeshType::SubdomainOverlapsType lowerOverlap, upperOverlap;
   SubdomainOverlapsGetter< MeshType, CommunicatorType >::getOverlaps( &distributedGrid, lowerOverlap, upperOverlap, 1 );
   distributedGrid.setOverlaps( lowerOverlap, upperOverlap );

   SharedPointer<MeshType> originalGrid;
   distributedGrid.setupGrid(*originalGrid);

   DofType dof(originalGrid->template getEntitiesCount< Cell >());
   dof.setValue(0); 

   SharedPointer<MeshFunction<MeshType>> meshFunctionptr;
   meshFunctionptr->bind(originalGrid,dof);

   MeshFunctionEvaluator< MeshFunction<MeshType>, LinearFunctionType > linearFunctionEvaluator;
   SharedPointer< LinearFunctionType, Host > linearFunctionPtr;
   linearFunctionEvaluator.evaluateAllEntities(meshFunctionptr , linearFunctionPtr);

   //Prepare Mesh Function parts for Cut 
   CutDistributedMeshType cutDistributedGrid;
   SharedPointer<CutMeshType> cutGrid;
   cutGrid->setDistMesh(&cutDistributedGrid);
   DofType cutDof(0);
   bool inCut=CutMeshFunction<CommunicatorType, MeshFunction<MeshType>,CutMeshType,DofType>::Cut(
            *meshFunctionptr,*cutGrid, cutDof, 
            StaticVector<2,int>(0,2),
            StaticVector<1,int>(1),
            StaticVector<1,typename CutMeshType::IndexType>(4) );

   if(inCut)
   {
       MeshFunction<CutMeshType> cutMeshFunction;
            cutMeshFunction.bind(cutGrid,cutDof); 

        cutMeshFunction.template synchronize<CommunicatorType>();

        typename MeshType::Cell fromEntity(meshFunctionptr->getMesh());
        typename CutMeshType::Cell outEntity(*cutGrid);

        for(int i=0;i<distributedGrid.getLocalGridSize().x();i++)
            for(int j=0;j<distributedGrid.getLocalGridSize().z();j++)
            {                
                fromEntity.getCoordinates().x()=i;
                fromEntity.getCoordinates().z()=j;
                fromEntity.getCoordinates().y()=4-distributedGrid.getGlobalBegin().y();
                outEntity.getCoordinates().x()=i;
                outEntity.getCoordinates().y()=j;
                fromEntity.refresh();
                outEntity.refresh();

                EXPECT_EQ(cutMeshFunction.getValue(outEntity), (*linearFunctionPtr)(fromEntity)) << "Error in Left overlap";

            }
      }
}


//=========================================================================================================
TEST(CutDistributedMeshFunction, 3D_2_Save)
{
   typedef Grid<3, double,Host,int> MeshType;
   typedef Grid<2, double,Host,int> CutMeshType;
   typedef DistributedMesh<MeshType> DistributedMeshType;
   typedef DistributedMesh<CutMeshType> CutDistributedMeshType;


   typedef Vector<double,Host,int> DofType;

   typedef typename MeshType::PointType PointType;
   typedef typename MeshType::Cell Cell;

   typedef LinearFunction<double,3> LinearFunctionType;

   MeshType globalOriginalGrid; 
   PointType origin;
   origin.setValue(-0.5);
   PointType proportions;
   proportions.setValue(10);
   globalOriginalGrid.setDimensions(proportions);
   globalOriginalGrid.setDomain(origin,proportions);

   typename DistributedMeshType::CoordinatesType overlap;
   overlap.setValue(1);
   DistributedMeshType distributedGrid;
   distributedGrid.setDomainDecomposition( typename DistributedMeshType::CoordinatesType( 2,2,3 ) );
   distributedGrid.template setGlobalGrid<CommunicatorType>( globalOriginalGrid );
   typename DistributedMeshType::SubdomainOverlapsType lowerOverlap, upperOverlap;
   SubdomainOverlapsGetter< MeshType, CommunicatorType >::getOverlaps( &distributedGrid, lowerOverlap, upperOverlap, 1 );
   distributedGrid.setOverlaps( lowerOverlap, upperOverlap );

   SharedPointer<MeshType> originalGrid;
   distributedGrid.setupGrid(*originalGrid);

   DofType dof(originalGrid->template getEntitiesCount< Cell >());
   dof.setValue(0); 

   SharedPointer<MeshFunction<MeshType>> meshFunctionptr;
   meshFunctionptr->bind(originalGrid,dof);

   MeshFunctionEvaluator< MeshFunction<MeshType>, LinearFunctionType > linearFunctionEvaluator;
   SharedPointer< LinearFunctionType, Host > linearFunctionPtr;
   linearFunctionEvaluator.evaluateAllEntities(meshFunctionptr , linearFunctionPtr);

   //Prepare Mesh Function parts for Cut 
   CutDistributedMeshType cutDistributedGrid;
   SharedPointer<CutMeshType> cutGrid;
   cutGrid->setDistMesh(&cutDistributedGrid);
   DofType cutDof(0);
   bool inCut=CutMeshFunction<CommunicatorType, MeshFunction<MeshType>,CutMeshType,DofType>::Cut(
            *meshFunctionptr,*cutGrid, cutDof, 
            StaticVector<2,int>(0,2),
            StaticVector<1,int>(1),
            StaticVector<1,typename CutMeshType::IndexType>(4) );

   
   String FileName=String("/tmp/test-file.tnl");
   if(inCut)
   {
       MeshFunction<CutMeshType> cutMeshFunction;
            cutMeshFunction.bind(cutGrid,cutDof); 
       
        DistributedGridIO<MeshFunction<CutMeshType>,MpiIO> ::save(FileName, cutMeshFunction );
        
        //save globalgrid for debug render
        typename CommunicatorType::CommunicationGroup *group;
        group=(typename CommunicatorType::CommunicationGroup *)(cutDistributedGrid.getCommunicationGroup());
        if(CommunicatorType::GetRank(*group)==0)
        {
            File meshFile;
            meshFile.open( FileName+String("-mesh.tnl"),IOMode::write);
            cutDistributedGrid.getGlobalGrid().save( meshFile );
            meshFile.close();
        }

    }

   if(CommunicatorType::GetRank(CommunicatorType::AllGroup)==0)
   {
       SharedPointer<CutMeshType> globalCutGrid;
       MeshFunction<CutMeshType> loadMeshFunctionptr;

       globalCutGrid->setDimensions(typename CutMeshType::CoordinatesType(10));
       globalCutGrid->setDomain(typename CutMeshType::PointType(-0.5),typename CutMeshType::CoordinatesType(10));

       DofType loaddof(globalCutGrid->template getEntitiesCount< typename CutMeshType::Cell >());
       loaddof.setValue(0); 
       loadMeshFunctionptr.bind(globalCutGrid,loaddof);

        File file;
        file.open( FileName, IOMode::read );
        loadMeshFunctionptr.boundLoad(file);
        file.close();
 
        typename MeshType::Cell fromEntity(globalOriginalGrid);
        typename CutMeshType::Cell outEntity(*globalCutGrid);

        for(int i=0;i<globalOriginalGrid.getDimensions().x();i++)
            for(int j=0;j<globalOriginalGrid.getDimensions().z();j++)
            {                
                fromEntity.getCoordinates().x()=i;
                fromEntity.getCoordinates().z()=j;
                fromEntity.getCoordinates().y()=4;
                outEntity.getCoordinates().x()=i;
                outEntity.getCoordinates().y()=j;
                fromEntity.refresh();
                outEntity.refresh();

                EXPECT_EQ(loadMeshFunctionptr.getValue(outEntity), (*linearFunctionPtr)(fromEntity)) << "Error in Left overlap";

            }
      }
  
}



#else
TEST(NoMPI, NoTest)
{
    ASSERT_TRUE(true) << ":-(";
}
#endif

#endif


#if (defined(HAVE_GTEST) && defined(HAVE_MPI))
#include <sstream>

  class MinimalistBufferedPrinter : public ::testing::EmptyTestEventListener {
      
  private:
      std::stringstream sout;
      
  public:
      
    // Called before a test starts.
    virtual void OnTestStart(const ::testing::TestInfo& test_info) {
      sout<< test_info.test_case_name() <<"." << test_info.name() << " Start." <<std::endl;
    }

    // Called after a failed assertion or a SUCCEED() invocation.
    virtual void OnTestPartResult(
        const ::testing::TestPartResult& test_part_result) {
      sout << (test_part_result.failed() ? "====Failure=== " : "===Success=== ") 
              << test_part_result.file_name() << " "
              << test_part_result.line_number() <<std::endl
              << test_part_result.summary() <<std::endl;
    }

    // Called after a test ends.
    virtual void OnTestEnd(const ::testing::TestInfo& test_info) 
    {
        int rank=CommunicatorType::GetRank(CommunicatorType::AllGroup);
        sout<< test_info.test_case_name() <<"." << test_info.name() << " End." <<std::endl;
        std::cout << rank << ":" << std::endl << sout.str()<< std::endl;
        sout.str( std::string() );
        sout.clear();
    }
  };
#endif

#include "../../src/UnitTests/GtestMissingError.h"
int main( int argc, char* argv[] )
{
#ifdef HAVE_GTEST
   ::testing::InitGoogleTest( &argc, argv );

    #ifdef HAVE_MPI
       ::testing::TestEventListeners& listeners =
          ::testing::UnitTest::GetInstance()->listeners();

       delete listeners.Release(listeners.default_result_printer());
       listeners.Append(new MinimalistBufferedPrinter);

       CommunicatorType::Init(argc,argv);
    #endif
       int result= RUN_ALL_TESTS();

    #ifdef HAVE_MPI
       CommunicatorType::Finalize();
    #endif
       return result;
#else
   
   throw GtestMissingError();
#endif
}

