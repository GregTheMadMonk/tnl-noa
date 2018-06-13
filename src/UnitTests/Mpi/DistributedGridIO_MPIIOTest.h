/***************************************************************************
                          DistributedGridIO_MPIIO  -  description
                             -------------------
    begin                : Nov 1, 2017
    copyright            : (C) 2017 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

#include <TNL/Communicators/MpiCommunicator.h>
#include <TNL/Meshes/DistributedMeshes/DistributedMesh.h>
#include <TNL/Functions/MeshFunction.h>

#ifdef HAVE_MPI
    #define MPIIO
#endif
#include <TNL/Meshes/DistributedMeshes/DistributedGridIO.h>


#include "Functions.h"

using namespace TNL::Containers;
using namespace TNL::Meshes;
using namespace TNL::Functions;
using namespace TNL::Devices;
using namespace TNL::Communicators;
using namespace TNL::Meshes::DistributedMeshes;

//------------------------------------------------------------------------------

typedef MpiCommunicator CommunicatorType;

template <int dim, typename Device>
class TestDistributedGridMPIIO{
    public:

    typedef Grid<dim,double,Device,int> MeshType;
    typedef MeshFunction<MeshType> MeshFunctionType;
    typedef Vector<double,Device,int> DofType;
    typedef typename MeshType::Cell Cell;
    typedef typename MeshType::IndexType IndexType; 
    typedef typename MeshType::PointType PointType; 
    typedef DistributedMesh<MeshType> DistributedGridType;

    typedef typename DistributedGridType::CoordinatesType CoordinatesType;
    typedef LinearFunction<double,dim> LinearFunctionType;

    static void TestSave()
    {
        SharedPointer< LinearFunctionType, Device > linearFunctionPtr;
        MeshFunctionEvaluator< MeshFunctionType, LinearFunctionType > linearFunctionEvaluator;    
        
        //save distributed meshfunction into file
        PointType globalOrigin;
        globalOrigin.setValue(-0.5);

        PointType globalProportions;
        globalProportions.setValue(50);

        SharedPointer<MeshType> globalGrid;
        globalGrid->setDimensions(globalProportions);
        globalGrid->setDomain(globalOrigin,globalProportions);
        
        CoordinatesType overlap;
        overlap.setValue(1);
        DistributedGridType distrgrid;
        distrgrid.template setGlobalGrid<CommunicatorType>( *globalGrid, overlap );

        ///std::cout << distrgrid.printProcessDistr() <<std::endl;

        SharedPointer<MeshType> gridptr;
        SharedPointer<MeshFunctionType> meshFunctionptr;
        distrgrid.setupGrid(*gridptr);
       
        DofType dof(gridptr->template getEntitiesCount< Cell >());
        dof.setValue(0);
        meshFunctionptr->bind(gridptr,dof);
            
        linearFunctionEvaluator.evaluateAllEntities(meshFunctionptr , linearFunctionPtr);
 
        String FileName=String("/tmp/test-file.tnl");
        DistributedGridIO<MeshFunctionType,MpiIO> ::save(FileName, *meshFunctionptr );

       //first process compare results
       if(CommunicatorType::GetRank()==0)
       {
            DofType globalEvaluatedDof(globalGrid->template getEntitiesCount< Cell >());

            SharedPointer<MeshFunctionType> globalEvaluatedMeshFunctionptr;
            globalEvaluatedMeshFunctionptr->bind(globalGrid,globalEvaluatedDof);
            linearFunctionEvaluator.evaluateAllEntities(globalEvaluatedMeshFunctionptr , linearFunctionPtr);


            DofType loadDof(globalGrid->template getEntitiesCount< Cell >());
            SharedPointer<MeshFunctionType> loadMeshFunctionptr;
            loadMeshFunctionptr->bind(globalGrid,loadDof);

            loadDof.setValue(-1);
        
            File file;
            file.open( FileName, IOMode::read );
            loadMeshFunctionptr->boundLoad(file);
            file.close();

            for(int i=0;i<loadDof.getSize();i++)
            {
              EXPECT_EQ( globalEvaluatedDof.getElement(i), loadDof.getElement(i)) << "Compare Loaded and evaluated Dof Failed for: "<< i;
            }
        }
    }
    
    static void TestLoad()
    {
        SharedPointer< LinearFunctionType, Device > linearFunctionPtr;
        MeshFunctionEvaluator< MeshFunctionType, LinearFunctionType > linearFunctionEvaluator;    

        //Crete distributed grid            
        PointType globalOrigin;
        globalOrigin.setValue(-0.5);

        PointType globalProportions;
        globalProportions.setValue(50);

        SharedPointer<MeshType> globalGrid;
        globalGrid->setDimensions(globalProportions);
        globalGrid->setDomain(globalOrigin,globalProportions);

        CoordinatesType overlap;
        overlap.setValue(1);
        DistributedGridType distrgrid;
        distrgrid.template setGlobalGrid<CommunicatorType>(*globalGrid,overlap);

        String FileName=String("/tmp/test-file.tnl");         

        //Prepare file   
        if(CommunicatorType::GetRank()==0)
        {   
            DofType saveDof(globalGrid->template getEntitiesCount< Cell >());

            SharedPointer<MeshFunctionType> saveMeshFunctionptr;
            saveMeshFunctionptr->bind(globalGrid,saveDof);
            linearFunctionEvaluator.evaluateAllEntities(saveMeshFunctionptr , linearFunctionPtr);
      
            File file;
            file.open( FileName, IOMode::write );        
            saveMeshFunctionptr->save(file);
            file.close();
        }

        SharedPointer<MeshType> loadGridptr;
        SharedPointer<MeshFunctionType> loadMeshFunctionptr;
        distrgrid.setupGrid(*loadGridptr);
        
        DofType loadDof(loadGridptr->template getEntitiesCount< Cell >());
        loadDof.setValue(0);
        loadMeshFunctionptr->bind(loadGridptr,loadDof);

        DistributedGridIO<MeshFunctionType,MpiIO> ::load(FileName, *loadMeshFunctionptr );
        loadMeshFunctionptr->template synchronize<CommunicatorType>(); //need synchronization for overlaps to be filled corectly in loadDof

        SharedPointer<MeshType> evalGridPtr;
        SharedPointer<MeshFunctionType> evalMeshFunctionptr;
        distrgrid.setupGrid(*evalGridPtr);
        
        DofType evalDof(evalGridPtr->template getEntitiesCount< Cell >());
        evalDof.setValue(-1);
        evalMeshFunctionptr->bind(evalGridPtr,evalDof);
        
        linearFunctionEvaluator.evaluateAllEntities(evalMeshFunctionptr , linearFunctionPtr);        
        evalMeshFunctionptr->template synchronize<CommunicatorType>();

        for(int i=0;i<evalDof.getSize();i++)
        {
            EXPECT_EQ( evalDof.getElement(i), loadDof.getElement(i)) << "Compare Loaded and evaluated Dof Failed for: "<< i;
        }
        
    }
};

