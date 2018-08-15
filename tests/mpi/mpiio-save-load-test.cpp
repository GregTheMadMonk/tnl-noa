#ifdef HAVE_MPI
#define MPIIO

#include <TNL/Communicators/MpiCommunicator.h>
#include <TNL/Functions/MeshFunction.h>
#include <TNL/Meshes/DistributedMeshes/DistributedMesh.h>
#include <TNL/Meshes/DistributedMeshes/DistributedGridIO.h>
#include <TNL/Meshes/DistributedMeshes/SubdomainOverlapsGetter.h>



#include "../../src/UnitTests/Functions/Functions.h"

#define DIM 1 

using namespace TNL::Containers;
using namespace TNL::Meshes;
using namespace TNL::Functions;
using namespace TNL::Devices;
using namespace TNL::Communicators;
using namespace TNL::Meshes::DistributedMeshes;


int main(int argc, char **argv)
{

        typedef Host Device;
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
        globalProportions.setValue(10);


        MeshType globalGrid;
        globalGrid.setDimensions(globalProportions);
        globalGrid.setDomain(globalOrigin,globalProportions);
        
        File meshFile;
        meshFile.open( String("globalGrid.tnl"),IOMode::write);
        globalGrid.save( meshFile );
        meshFile.close();

        DistributedGridType distributedGrid;
        distributedGrid.template setGlobalGrid<CommunicatorType>( globalGrid );
        typename DistributedGridType::SubdomainOverlapsType lowerOverlap, upperOverlap;
        SubdomainOverlapsGetter< MeshType, CommunicatorType >::getOverlaps( &distributedGrid, lowerOverlap, upperOverlap, 1 );
        distributedGrid.setOverlaps( lowerOverlap, upperOverlap );

        SharedPointer<MeshType> gridptr;
        SharedPointer<MeshFunctionType> meshFunctionptr;
        distributedGrid.setupGrid(*gridptr);
       
        DofType dofsave(gridptr->template getEntitiesCount< Cell >());
        dofsave.setValue(0);
        meshFunctionptr->bind(gridptr,dofsave);
            
        linearFunctionEvaluator.evaluateAllEntities(meshFunctionptr , linearFunctionPtr);
        
        String fileName=String("./meshFunction.tnl");
        DistributedGridIO<MeshFunctionType,MpiIO> ::save(fileName, *meshFunctionptr );

        DofType dofload(gridptr->template getEntitiesCount< Cell >());
        dofload.setValue(0);
        meshFunctionptr->bind(gridptr,dofload);

        DistributedGridIO<MeshFunctionType,MpiIO> ::load(fileName, *meshFunctionptr );

        for(int i=0;i<dofload.getSize();i++)
        {
            if(dofsave[i]!=dofload[i])
                std::cout<<"Chyba na pozici "<< i << " dofsave: "<< dofsave[i] << " dofload: "<<dofload[i] <<std::endl;
            else
                std::cout <<"Ok!"<<std::endl;
        }

        CommunicatorType::Finalize();

}

#else

main(){}
#endif

