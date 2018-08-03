#include <TNL/Communicators/MpiCommunicator.h>
#include <TNL/Meshes/DistributedMeshes/DistributedMesh.h>
#include <TNL/Functions/MeshFunction.h>
#include <TNL/Functions/VectorField.h>

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

template <int dim, int vctdim, typename Device>
class TestDistributedVectorFieldMPIIO{
    public:

    typedef Grid<dim,double,Device,int> MeshType;
    typedef MeshFunction<MeshType> MeshFunctionType;
	typedef VectorField<vctdim,MeshFunctionType> VectorFieldType;
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
        globalProportions.setValue(10);

        SharedPointer<MeshType> globalGrid;
        globalGrid->setDimensions(globalProportions);
        globalGrid->setDomain(globalOrigin,globalProportions);
        
        CoordinatesType overlap;
        overlap.setValue(1);
        DistributedGridType distrgrid;
        distrgrid.template setGlobalGrid<CommunicatorType>( *globalGrid, overlap );

        ///std::cout << distrgrid.printProcessDistr() <<std::endl;

        SharedPointer<MeshType> gridptr;        
        distrgrid.setupGrid(*gridptr);

		VectorFieldType vectorField;

        DofType dof(vctdim*(gridptr->template getEntitiesCount< Cell >()));
        dof.setValue(0);
        vectorField.bind(gridptr,dof);
            
		for(int i=0;i<vctdim;i++)
	        linearFunctionEvaluator.evaluateAllEntities(vectorField [ i ], linearFunctionPtr);
 
        String FileName=String("/tmp/test-file.tnl");
        DistributedGridIO<VectorFieldType,MpiIO> ::save(FileName, vectorField );
        /*File file;
        file.open( FileName, IOMode::write );
		vectorField.save(file);
		file.close();		*/

       //first process compare results
       if(CommunicatorType::GetRank(CommunicatorType::AllGroup)==0)
       {
            DofType globalEvaluatedDof(vctdim*(globalGrid->template getEntitiesCount< Cell >()));

            VectorFieldType globalEvaluatedvct;
            globalEvaluatedvct.bind(globalGrid,globalEvaluatedDof);
			for(int i=0;i<vctdim;i++)
	            linearFunctionEvaluator.evaluateAllEntities(globalEvaluatedvct[i] , linearFunctionPtr);


            DofType loadDof(vctdim*(globalGrid->template getEntitiesCount< Cell >()));
            VectorFieldType loadvct;
            loadvct.bind(globalGrid,loadDof);

            loadDof.setValue(-1);
        
            File file;
            file.open( FileName, IOMode::read );
			bool loaded=loadvct.boundLoad(file);
			file.close();
            if(!loaded)
				EXPECT_TRUE(loaded)<< "Chyba načtení souboru" <<std::endl;
            else
		        for(int i=0;i<loadDof.getSize();i++)
		        {
		          EXPECT_EQ( globalEvaluatedDof.getElement(i), loadDof.getElement(i)) << "Compare Loaded and evaluated Dof Failed for: "<< i;
		        }
		    }
    };
    
/*    static void TestLoad()
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
        if(CommunicatorType::GetRank(CommunicatorType::AllGroup)==0)
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
        
    }*/
};
