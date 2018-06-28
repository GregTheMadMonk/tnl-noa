#include <iostream>

#include <TNL/Functions/MeshFunction.h>
//#include <TNL/Devices/Host.h>

#include "../../src/UnitTests/Mpi/Functions.h"


using namespace std;
using namespace TNL;
using namespace TNL::Functions;
using namespace TNL::Meshes;
using namespace TNL::Devices;
using namespace TNL::Containers;

//template< typename Device >
struct For
{
    template < typename Index,
             typename Function,
             typename... FunctionArgs,
             int dim>
    static void exec( StaticVector<dim,Index> starts, StaticVector<dim,Index> ends, Function f, FunctionArgs... args )
    {
        StaticVector<dim,Index> index;
        if(dim==1)
        {
            for(index[0]=starts[0]; index[0]< ends[0];index[0]++ )
                 f( index, args... );
        }

        if(dim==2)
        {
            for(index[1]=starts[1]; index[1]< ends[1];index[1]++ )
                for(index[0]=starts[0]; index[0]< ends[0];index[0]++ )
                        f( index, args... );
        }

        if(dim==3)
        {
            for(index[2]=starts[2]; index[2]< ends[2];index[2]++ )
                for(index[1]=starts[1]; index[1]< ends[1];index[1]++ )
                    for(index[0]=starts[0]; index[0]< ends[0];index[0]++ )
                        f( index, args... );
        }
    }
};

template <  typename MeshFunctionType,
            typename OutMesh,
            typename OutDof,
            int outDimension=OutMesh::getMeshDimension(),
            int codimension=MeshFunctionType::getMeshDimension()-OutMesh::getMeshDimension()>
class CutMeshFunction
{
  public:
    static void Cut(MeshFunctionType &inputMeshFunction, OutMesh &outMesh, OutDof &outData,StaticVector<outDimension, int> savedDimensions, StaticVector<codimension,int> reducedDimensions, StaticVector<codimension,typename MeshFunctionType::IndexType> fixedIndexs )
    {
        auto fromData=inputMeshFunction.getData().getData();
        auto fromMesh=inputMeshFunction.getMesh();
    
        //Set up output structures
        typename OutMesh::PointType outOrigin;
        typename OutMesh::PointType outProportions;
        typename OutMesh::CoordinatesType outDimensions;
        
        for(int i=0; i<outDimension;i++)
        {
            outOrigin[i]=fromMesh.getOrigin()[savedDimensions[i]];
            outProportions[i]=fromMesh.getProportions()[savedDimensions[i]];
            outDimensions[i]=fromMesh.getDimensions()[savedDimensions[i]];
        }
        
        outMesh.setDimensions(outDimensions);
        outMesh.setDomain(outOrigin,outProportions);
        outData.setSize(outMesh.template getEntitiesCount< typename OutMesh::Cell >());

        //copy data
        auto kernel = [&fromData, &fromMesh, &outData, &outMesh, &savedDimensions,&fixedIndexs,&reducedDimensions ] ( typename OutMesh::CoordinatesType index )
        {
            typename MeshFunctionType::MeshType::Cell fromEntity(fromMesh);
            typename OutMesh::Cell outEntity(outMesh);

            for(int j=0;j<outDimension;j++)
            {
                fromEntity.getCoordinates()[savedDimensions[j]]=index[j];
                outEntity.getCoordinates()[j]=index[j];
            }

            for(int j=0; j<codimension;j++)
                fromEntity.getCoordinates()[reducedDimensions[j]]=fixedIndexs[j];

            fromEntity.refresh();
            outEntity.refresh();
            outData[outEntity.getIndex()]=fromData[fromEntity.getIndex()];
        };


        typename OutMesh::CoordinatesType starts;
        starts.setValue(0);
        For::exec(starts,outDimensions,kernel);
    } 
};



#define DIM 3
#define CODIM 1
#define DEVICE Host
#define SIZE 5


int main(int argc, char **argv)
{
   typedef Grid<DIM, double,DEVICE,int> MeshType;
   typedef Grid<DIM-CODIM, double,DEVICE,int> CutMeshType;
   typedef Vector<double,DEVICE,int> DofType;

   typedef typename MeshType::PointType PointType;
   typedef typename MeshType::Cell Cell;

   typedef LinearFunction<double,DIM> LinearFunctionType;
  

   //Original MeshFunciton --filed with linear function
   SharedPointer<MeshType> originalGrid;
   SharedPointer<MeshFunction<MeshType>> meshFunctionptr;
 
   PointType origin;
   origin.setValue(-0.5);
   PointType proportions;
   proportions.setValue(SIZE);
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


   for(int i=0;i<SIZE;i++)
   {
        CutMeshFunction<MeshFunction<MeshType>,CutMeshType,DofType>::Cut(
            *meshFunctionptr,*cutGrid, cutDof, 
            StaticVector<DIM-CODIM,int>(0,2),
            StaticVector<CODIM,int>(1),
            StaticVector<CODIM,typename CutMeshType::IndexType>(i) );

        MeshFunction<CutMeshType> cutMeshFunction;
        cutMeshFunction.bind(cutGrid,cutDof);  
        
        Printer<CutMeshType,DofType>::print_dof(0,*cutGrid, cutDof);

   }

   

  return 0;
}
