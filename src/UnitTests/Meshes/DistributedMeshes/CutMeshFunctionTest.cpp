#ifdef HAVE_GTEST  
#include <gtest/gtest.h>

#include <TNL/Functions/CutMeshFunction.h>
#include <TNL/Devices/Host.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/Communicators/NoDistrCommunicator.h>

#include "../../Functions/Functions.h"

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Functions;
using namespace TNL::Meshes;
using namespace TNL::Devices;
using namespace TNL::Communicators;


TEST(CutMeshFunction, 2D)
{

   typedef Grid<2, double,Host,int> MeshType;
   typedef Grid<1, double,Host,int> CutMeshType;
   typedef Vector<double,Host,int> DofType;

   typedef typename MeshType::PointType PointType;
   typedef typename MeshType::Cell Cell;

   typedef LinearFunction<double,2> LinearFunctionType;
  

   //Original MeshFunciton --filed with linear function
   SharedPointer<MeshType> originalGrid;
   SharedPointer<MeshFunction<MeshType>> meshFunctionptr;
 
   PointType origin;
   origin.setValue(-0.5);
   PointType proportions;
   proportions.setValue(10);
   originalGrid->setDimensions(proportions);
   originalGrid->setDomain(origin,proportions);


   DofType dof(originalGrid->template getEntitiesCount< Cell >());
   dof.setValue(0); 
   meshFunctionptr->bind(originalGrid,dof);

   MeshFunctionEvaluator< MeshFunction<MeshType>, LinearFunctionType > linearFunctionEvaluator;
   SharedPointer< LinearFunctionType, Host > linearFunctionPtr;
   linearFunctionEvaluator.evaluateAllEntities(meshFunctionptr , linearFunctionPtr);
 
   //Prepare Mesh Function parts for Cut 
   SharedPointer<CutMeshType> cutGrid;
   DofType cutDof(0);
   bool inCut=CutMeshFunction<NoDistrCommunicator,MeshFunction<MeshType>,CutMeshType,DofType>::Cut(
            *meshFunctionptr,*cutGrid, cutDof, 
            StaticVector<1,int>(0),
            StaticVector<1,int>(1),
            StaticVector<1,typename CutMeshType::IndexType>(5) );

   ASSERT_TRUE(inCut)<<"nedistribuovaná meshfunction musí být vždy v řezu";

   MeshFunction<CutMeshType> cutMeshFunction;
        cutMeshFunction.bind(cutGrid,cutDof); 

    for(int i=0;i<10;i++)
    {
       typename MeshType::Cell fromEntity(meshFunctionptr->getMesh());
       typename CutMeshType::Cell outEntity(*cutGrid);
       
        fromEntity.getCoordinates().x()=i;
        fromEntity.getCoordinates().y()=5;
        outEntity.getCoordinates().x()=i;

        fromEntity.refresh();
        outEntity.refresh();

        EXPECT_EQ(cutDof[outEntity.getIndex()],dof[fromEntity.getIndex()]) <<"Chyba";
    }
}


TEST(CutMeshFunction, 3D_1)
{
   typedef Grid<3, double,Host,int> MeshType;
   typedef Grid<1, double,Host,int> CutMeshType;
   typedef Vector<double,Host,int> DofType;

   typedef typename MeshType::PointType PointType;
   typedef typename MeshType::Cell Cell;

   typedef LinearFunction<double,3> LinearFunctionType;
  

   //Original MeshFunciton --filed with linear function
   SharedPointer<MeshType> originalGrid;
   SharedPointer<MeshFunction<MeshType>> meshFunctionptr;
 
   PointType origin;
   origin.setValue(-0.5);
   PointType proportions;
   proportions.setValue(10);
   originalGrid->setDimensions(proportions);
   originalGrid->setDomain(origin,proportions);


   DofType dof(originalGrid->template getEntitiesCount< Cell >());
   dof.setValue(0); 
   meshFunctionptr->bind(originalGrid,dof);

   MeshFunctionEvaluator< MeshFunction<MeshType>, LinearFunctionType > linearFunctionEvaluator;
   SharedPointer< LinearFunctionType, Host > linearFunctionPtr;
   linearFunctionEvaluator.evaluateAllEntities(meshFunctionptr , linearFunctionPtr);
 
   //Prepare Mesh Function parts for Cut 
   SharedPointer<CutMeshType> cutGrid;
   DofType cutDof(0);
   bool inCut=CutMeshFunction<NoDistrCommunicator,MeshFunction<MeshType>,CutMeshType,DofType>::Cut(
            *meshFunctionptr,*cutGrid, cutDof, 
            StaticVector<1,int>(1),
            StaticVector<2,int>(0,2),
            StaticVector<2,typename CutMeshType::IndexType>(5,5) );

   ASSERT_TRUE(inCut)<<"nedistribuovaná meshfunction musí být vždy v řezu";

   MeshFunction<CutMeshType> cutMeshFunction;
        cutMeshFunction.bind(cutGrid,cutDof); 

    for(int i=0;i<10;i++)
    {
       typename MeshType::Cell fromEntity(meshFunctionptr->getMesh());
       typename CutMeshType::Cell outEntity(*cutGrid);
       
        fromEntity.getCoordinates().x()=5;
        fromEntity.getCoordinates().y()=i;
        fromEntity.getCoordinates().z()=5;
        outEntity.getCoordinates().x()=i;

        fromEntity.refresh();
        outEntity.refresh();

        EXPECT_EQ(cutDof[outEntity.getIndex()],dof[fromEntity.getIndex()]) <<"Chyba";
    }
}

TEST(CutMeshFunction, 3D_2)
{
   typedef Grid<3, double,Host,int> MeshType;
   typedef Grid<2, double,Host,int> CutMeshType;
   typedef Vector<double,Host,int> DofType;

   typedef typename MeshType::PointType PointType;
   typedef typename MeshType::Cell Cell;

   typedef LinearFunction<double,3> LinearFunctionType;
  

   //Original MeshFunciton --filed with linear function
   SharedPointer<MeshType> originalGrid;
   SharedPointer<MeshFunction<MeshType>> meshFunctionptr;
 
   PointType origin;
   origin.setValue(-0.5);
   PointType proportions;
   proportions.setValue(10);
   originalGrid->setDimensions(proportions);
   originalGrid->setDomain(origin,proportions);


   DofType dof(originalGrid->template getEntitiesCount< Cell >());
   dof.setValue(0); 
   meshFunctionptr->bind(originalGrid,dof);

   MeshFunctionEvaluator< MeshFunction<MeshType>, LinearFunctionType > linearFunctionEvaluator;
   SharedPointer< LinearFunctionType, Host > linearFunctionPtr;
   linearFunctionEvaluator.evaluateAllEntities(meshFunctionptr , linearFunctionPtr);
 
   //Prepare Mesh Function parts for Cut 
   SharedPointer<CutMeshType> cutGrid;
   DofType cutDof(0);
   bool inCut=CutMeshFunction<NoDistrCommunicator, MeshFunction<MeshType>,CutMeshType,DofType>::Cut(
            *meshFunctionptr,*cutGrid, cutDof, 
            StaticVector<2,int>(2,1),
            StaticVector<1,int>(0),
            StaticVector<1,typename CutMeshType::IndexType>(5) );

   ASSERT_TRUE(inCut)<<"nedistribuovaná meshfunction musí být vždy v řezu";

   MeshFunction<CutMeshType> cutMeshFunction;
        cutMeshFunction.bind(cutGrid,cutDof); 

    for(int i=0;i<10;i++)
    {
       for(int j=0;j<10;j++)
        {
           typename MeshType::Cell fromEntity(meshFunctionptr->getMesh());
           typename CutMeshType::Cell outEntity(*cutGrid);
           
            fromEntity.getCoordinates().x()=5;
            fromEntity.getCoordinates().y()=j;
            fromEntity.getCoordinates().z()=i;
            outEntity.getCoordinates().x()=i;
            outEntity.getCoordinates().y()=j;

            fromEntity.refresh();
            outEntity.refresh();

            EXPECT_EQ(cutDof[outEntity.getIndex()],dof[fromEntity.getIndex()]) <<i <<" "<< j <<"Chyba";
        }
    }
}


#endif

#include "../../src/UnitTests/GtestMissingError.h"
int main( int argc, char* argv[] )
{
#ifdef HAVE_GTEST
   ::testing::InitGoogleTest( &argc, argv );
       int result= RUN_ALL_TESTS();
       return result;
#else
   
   throw GtestMissingError();
#endif
}
