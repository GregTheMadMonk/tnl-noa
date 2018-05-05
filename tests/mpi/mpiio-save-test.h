#include <TNL/Communicators/MpiCommunicator.h>
#include <TNL/Meshes/DistributedMeshes/DistributedMesh.h>
#include <TNL/Functions/MeshFunction.h>

#define MPIIO
#include <TNL/Meshes/DistributedMeshes/DistributedGridIO.h>


#include "../../src/UnitTests/Mpi/Functions.h"

#define DIM 2

using namespace TNL::Containers;
using namespace TNL::Meshes;
using namespace TNL::Functions;
using namespace TNL::Devices;
using namespace TNL::Communicators;
using namespace TNL::Meshes::DistributedMeshes;


int main(int argc, char **argv)
{

        typedef DEVICE Device;
        typedef MpiCommunicator CommunicatorType;

        typedef Grid<DIM,double,Device,int> MeshType;
        typedef MeshFunction<MeshType> MeshFunctionType;
        typedef Vector<double,Device,int> DofType;
        typedef typename MeshType::Cell Cell;
        typedef typename MeshType::IndexType IndexType; 
        typedef typename MeshType::PointType PointType; 
        typedef DistributedMesh<MeshType> DistributedGridType;

        typedef typename DistributedGridType::CoordinatesType CoordinatesType;
        typedef LinearFunction<double,DIM> LinearFunctionType;

        CommunicatorType::Init(argc, argv);

        SharedPointer< LinearFunctionType, Device > linearFunctionPtr;
        MeshFunctionEvaluator< MeshFunctionType, LinearFunctionType > linearFunctionEvaluator;    
                
        //save distributed meshfunction into files
        PointType globalOrigin;
        globalOrigin.setValue(-0.5);

        PointType globalProportions;
        //globalProportions.setValue(10);
        globalProportions.x()=10;   
        globalProportions.y()=20;   
        


        MeshType globalGrid;
        globalGrid.setDimensions(globalProportions);
        globalGrid.setDomain(globalOrigin,globalProportions);
        
        File meshFile;
        meshFile.open( String("globalGrid.tnl"),IOMode::write);
        globalGrid.save( meshFile );
        meshFile.close();

        CoordinatesType overlap;
        overlap.setValue(1);
        DistributedGridType distrgrid;
        distrgrid.template setGlobalGrid<CommunicatorType>(globalGrid,overlap);

        SharedPointer<MeshType> gridptr;
        SharedPointer<MeshFunctionType> meshFunctionptr;
        distrgrid.SetupGrid(*gridptr);
       
        DofType dof(gridptr->template getEntitiesCount< Cell >());
        dof.setValue(0);
        meshFunctionptr->bind(gridptr,dof);
            
        linearFunctionEvaluator.evaluateAllEntities(meshFunctionptr , linearFunctionPtr);
        
        String fileName=String("./meshFunction.tnl");
        DistributedGridIO<MeshFunctionType,MpiIO> ::save(fileName, *meshFunctionptr );

        CommunicatorType::Finalize();

}
