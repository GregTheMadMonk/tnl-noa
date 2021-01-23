/***************************************************************************
                          DistributedGridIO_MPIIO  -  description
                             -------------------
    begin                : Nov 1, 2017
    copyright            : (C) 2017 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/


#ifdef HAVE_MPI
    #define MPIIO
#endif

#include <TNL/Meshes/DistributedMeshes/DistributedMeshSynchronizer.h>
#include <TNL/Meshes/DistributedMeshes/DistributedGridIO.h>
#include <TNL/Meshes/DistributedMeshes/SubdomainOverlapsGetter.h>
#include <TNL/Meshes/DistributedMeshes/DistributedMesh.h>
#include <TNL/Functions/MeshFunctionView.h>


#include "../../Functions/Functions.h"

using namespace TNL::Containers;
using namespace TNL::Meshes;
using namespace TNL::Functions;
using namespace TNL::Meshes::DistributedMeshes;

//------------------------------------------------------------------------------

template <int dim, typename Device>
class TestDistributedGridMPIIO{
    public:

    typedef Grid<dim,double,Device,int> MeshType;
    typedef MeshFunctionView<MeshType> MeshFunctionType;
    typedef Vector<double,Device,int> DofType;
    typedef typename MeshType::Cell Cell;
    typedef typename MeshType::IndexType IndexType;
    typedef typename MeshType::PointType PointType;
    typedef DistributedMesh<MeshType> DistributedGridType;

    typedef typename DistributedGridType::CoordinatesType CoordinatesType;
    typedef LinearFunction<double,dim> LinearFunctionType;

    static void TestSave()
    {
        Pointers::SharedPointer< LinearFunctionType, Device > linearFunctionPtr;
        MeshFunctionEvaluator< MeshFunctionType, LinearFunctionType > linearFunctionEvaluator;

        //save distributed meshfunction into file
        PointType globalOrigin;
        globalOrigin.setValue(-0.5);

        PointType globalProportions;
        globalProportions.setValue(50);

        Pointers::SharedPointer<MeshType> globalGrid;
        globalGrid->setDimensions(globalProportions);
        globalGrid->setDomain(globalOrigin,globalProportions);

        DistributedGridType distributedGrid;
        distributedGrid.setGlobalGrid( *globalGrid );
        typename DistributedGridType::SubdomainOverlapsType lowerOverlap, upperOverlap;
        SubdomainOverlapsGetter< MeshType >::getOverlaps( &distributedGrid, lowerOverlap, upperOverlap, 1 );
        distributedGrid.setOverlaps( lowerOverlap, upperOverlap );

        ///std::cout << distributedGrid.printProcessDistr() <<std::endl;

        Pointers::SharedPointer<MeshType> gridptr;
        Pointers::SharedPointer<MeshFunctionType> meshFunctionptr;
        distributedGrid.setupGrid(*gridptr);

        DofType dof(gridptr->template getEntitiesCount< Cell >());
        dof.setValue(0);
        meshFunctionptr->bind(gridptr,dof);

        linearFunctionEvaluator.evaluateAllEntities(meshFunctionptr , linearFunctionPtr);

        String FileName=String("test-file-mpiio-save.tnl");
        DistributedGridIO<MeshFunctionType,MpiIO> ::save(FileName, *meshFunctionptr );

       //first process compare results
       if(TNL::MPI::GetRank()==0)
       {
            DofType globalEvaluatedDof(globalGrid->template getEntitiesCount< Cell >());

            Pointers::SharedPointer<MeshFunctionType> globalEvaluatedMeshFunctionptr;
            globalEvaluatedMeshFunctionptr->bind(globalGrid,globalEvaluatedDof);
            linearFunctionEvaluator.evaluateAllEntities(globalEvaluatedMeshFunctionptr , linearFunctionPtr);


            DofType loadDof(globalGrid->template getEntitiesCount< Cell >());
            Pointers::SharedPointer<MeshFunctionType> loadMeshFunctionptr;
            loadMeshFunctionptr->bind(globalGrid,loadDof);

            loadDof.setValue(-1);

            File file;
            file.open( FileName, std::ios_base::in );
            loadMeshFunctionptr->boundLoad(file);
            file.close();

            for(int i=0;i<loadDof.getSize();i++)
            {
              EXPECT_EQ( globalEvaluatedDof.getElement(i), loadDof.getElement(i)) << "Compare Loaded and evaluated Dof Failed for: "<< i;
            }
            EXPECT_EQ( std::remove( FileName.getString()) , 0 );
        }
    }

    static void TestLoad()
    {
        Pointers::SharedPointer< LinearFunctionType, Device > linearFunctionPtr;
        MeshFunctionEvaluator< MeshFunctionType, LinearFunctionType > linearFunctionEvaluator;

        //Crete distributed grid
        PointType globalOrigin;
        globalOrigin.setValue(-0.5);

        PointType globalProportions;
        globalProportions.setValue(50);

        Pointers::SharedPointer<MeshType> globalGrid;
        globalGrid->setDimensions(globalProportions);
        globalGrid->setDomain(globalOrigin,globalProportions);

        CoordinatesType overlap;
        overlap.setValue(1);
        DistributedGridType distributedGrid;
        distributedGrid.setGlobalGrid( *globalGrid );
        typename DistributedGridType::SubdomainOverlapsType lowerOverlap, upperOverlap;
        SubdomainOverlapsGetter< MeshType >::getOverlaps( &distributedGrid, lowerOverlap, upperOverlap, 1 );
        distributedGrid.setOverlaps( lowerOverlap, upperOverlap );

        String FileName=String("test-file-mpiio-load.tnl");

        //Prepare file
        if(TNL::MPI::GetRank()==0)
        {
            DofType saveDof(globalGrid->template getEntitiesCount< Cell >());

            Pointers::SharedPointer<MeshFunctionType> saveMeshFunctionptr;
            saveMeshFunctionptr->bind(globalGrid,saveDof);
            linearFunctionEvaluator.evaluateAllEntities(saveMeshFunctionptr , linearFunctionPtr);

            File file;
            file.open( FileName, std::ios_base::out );
            saveMeshFunctionptr->save(file);
            file.close();
        }

        Pointers::SharedPointer<MeshType> loadGridptr;
        Pointers::SharedPointer<MeshFunctionType> loadMeshFunctionptr;
        distributedGrid.setupGrid(*loadGridptr);

        DofType loadDof(loadGridptr->template getEntitiesCount< Cell >());
        loadDof.setValue(0);
        loadMeshFunctionptr->bind(loadGridptr,loadDof);

        DistributedGridIO<MeshFunctionType,MpiIO> ::load(FileName, *loadMeshFunctionptr );

        DistributedMeshSynchronizer< DistributedGridType > synchronizer;
        synchronizer.setDistributedGrid( &distributedGrid );
        synchronizer.synchronize( *loadMeshFunctionptr ); //need synchronization for overlaps to be filled corectly in loadDof

        Pointers::SharedPointer<MeshType> evalGridPtr;
        Pointers::SharedPointer<MeshFunctionType> evalMeshFunctionptr;
        distributedGrid.setupGrid(*evalGridPtr);

        DofType evalDof(evalGridPtr->template getEntitiesCount< Cell >());
        evalDof.setValue(-1);
        evalMeshFunctionptr->bind(evalGridPtr,evalDof);

        linearFunctionEvaluator.evaluateAllEntities(evalMeshFunctionptr , linearFunctionPtr);
        synchronizer.synchronize( *evalMeshFunctionptr );

        for(int i=0;i<evalDof.getSize();i++)
        {
            EXPECT_EQ( evalDof.getElement(i), loadDof.getElement(i)) << "Compare Loaded and evaluated Dof Failed for: "<< i;
        }

        if(TNL::MPI::GetRank()==0)
        {
            EXPECT_EQ( std::remove( FileName.getString()) , 0 );
        }

    }
};
