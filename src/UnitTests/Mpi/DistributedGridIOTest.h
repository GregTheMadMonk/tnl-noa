/***************************************************************************
                          DistributedGridIO.h  -  description
                             -------------------
    begin                : Nov 1, 2017
    copyright            : (C) 2017 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

#include <TNL/Communicators/MpiCommunicator.h>
#include <TNL/Meshes/DistributedMeshes/DistributedMesh.h>
#include <TNL/Functions/MeshFunction.h>
#include <TNL/Meshes/DistributedMeshes/DistributedGridIO.h>


#include "Functions.h"

using namespace TNL::Containers;
using namespace TNL::Meshes;
using namespace TNL::Functions;
using namespace TNL::Devices;
using namespace TNL::Communicators;
using namespace TNL::Meshes::DistributedMeshes;


//================Parameters===================================
template <int dim, typename Device>
class ParameterProvider
{
    public:

    typedef Grid<dim,double,Device,int> MeshType;
    typedef typename MeshType::CoordinatesType CoordinatesType;
    typedef typename MeshType::PointType PointType;

    PointType getOrigin(int rank)
    {
    };

    PointType getProportions(int rank)
    {
    };

    int* getDistr(void)
    {
        return NULL;
    };
};

template<typename Device>
class ParameterProvider<1,Device>
{
    public:

    typedef Grid<1,double,Device,int> MeshType;
    typedef typename MeshType::CoordinatesType CoordinatesType;
    typedef typename MeshType::PointType PointType;

    PointType getOrigin(int rank)
    {
        if(rank==0)
            return PointType(-0.5);
        if(rank==1)
            return PointType(4.5);
        if(rank==2)
            return PointType(9.5);
        if(rank==3)
            return PointType(14.5);

        return PointType(0);
    };

    PointType getProportions(int rank)
    {
        if(rank==0)
            return PointType(5);
        if(rank==1)
            return PointType(5);
        if(rank==2)
            return PointType(5);
        if(rank==3)
            return PointType(5);
        return PointType(0);
    };

    const CoordinatesType& getDistr()
    {
        distr[0]=4;
        return distr;
    };
    
    CoordinatesType distr;
};

template<typename Device>
class ParameterProvider<2,Device>
{
    public:

    typedef Grid<2,double,Device,int> MeshType;
    typedef typename MeshType::CoordinatesType CoordinatesType;
    typedef typename MeshType::PointType PointType;

    PointType getOrigin(int rank)
    {
        if(rank==0)
            return PointType(-0.5,-0.5);
        if(rank==1)
            return PointType(9.5,-0.5);
        if(rank==2)
            return PointType(-0.5,9.5);
        if(rank==3)
            return PointType(9.5,9.5);

        return PointType(0,0);
    };

    PointType getProportions(int rank)
    {
        if(rank==0)
            return PointType(10,10);
        if(rank==1)
            return PointType(10,10);
        if(rank==2)
            return PointType(10,10);
        if(rank==3)
            return PointType(10,10);
        return PointType(0,0);
    };

    const CoordinatesType& getDistr()
    {
        distr[0]=2;
        distr[1]=2;
        return distr;
    };
    
    CoordinatesType distr;
};

template<typename Device>
class ParameterProvider<3,Device>
{
    public:

    typedef Grid<3,double,Device,int> MeshType;
    typedef typename MeshType::CoordinatesType CoordinatesType;
    typedef typename MeshType::PointType PointType;

    PointType getOrigin(int rank)
    {
        if(rank==0)
            return PointType(-0.5,-0.5,-0.5);
        if(rank==1)
            return PointType(9.5,-0.5,-0.5);
        if(rank==2)
            return PointType(-0.5,9.5,-0.5);
        if(rank==3)
            return PointType(9.5,9.5,-0.5);

        return PointType(0,0,0);
    };

    PointType getProportions(int rank)
    {
        if(rank==0)
            return PointType(10,10,20);
        if(rank==1)
            return PointType(10,10,20);
        if(rank==2)
            return PointType(10,10,20);
        if(rank==3)
            return PointType(10,10,20);
        return PointType(0,0,0);
    };

    const CoordinatesType& getDistr()
    {
        distr[0]=2;
        distr[1]=2;
        distr[2]=1;
        return distr;
    };
    
    CoordinatesType distr;
};

//------------------------------------------------------------------------------

typedef MpiCommunicator CommunicatorType;

template <int dim, typename Device>
class TestDistributedGridIO{
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
        
        ParameterProvider<dim,Device> parameters;
        
        //save distributed meshfunction into files
        PointType globalOrigin;
        globalOrigin.setValue(-0.5);

        PointType globalProportions;
        globalProportions.setValue(20);


        MeshType globalGrid;
        globalGrid.setDimensions(globalProportions);
        globalGrid.setDomain(globalOrigin,globalProportions);
        
        CoordinatesType overlap;
        overlap.setValue(1);
        DistributedGridType distrgrid;
        distrgrid.setDomainDecomposition( parameters.getDistr() );
        distrgrid.template setGlobalGrid<CommunicatorType>( globalGrid, overlap );

        std::cout << distrgrid.printProcessDistr() <<std::endl;

        SharedPointer<MeshType> gridptr;
        SharedPointer<MeshFunctionType> meshFunctionptr;
        distrgrid.setupGrid(*gridptr);
       
        DofType dof(gridptr->template getEntitiesCount< Cell >());
        dof.setValue(0);
        meshFunctionptr->bind(gridptr,dof);
            
        linearFunctionEvaluator.evaluateAllEntities(meshFunctionptr , linearFunctionPtr);
 
        String FileName=String("/tmp/test-file.tnl");
        DistributedGridIO<MeshFunctionType> ::save(FileName, *meshFunctionptr );


       //create similar local mesh function and evaluate linear function on it
        PointType localOrigin=parameters.getOrigin(CommunicatorType::GetRank());        
        PointType localProportions=parameters.getProportions(CommunicatorType::GetRank());;
            
        SharedPointer<MeshType>  localGridptr;
        localGridptr->setDimensions(localProportions);
        localGridptr->setDomain(localOrigin,localProportions);

        DofType localDof(localGridptr->template getEntitiesCount< Cell >());

        SharedPointer<MeshFunctionType> localMeshFunctionptr;
        localMeshFunctionptr->bind(localGridptr,localDof);
        linearFunctionEvaluator.evaluateAllEntities(localMeshFunctionptr , linearFunctionPtr);

        //load other meshfunction on same localgrid from created file
        SharedPointer<MeshType>  loadGridptr;
        loadGridptr->setDimensions(localProportions);
        loadGridptr->setDomain(localOrigin,localProportions);

        DofType loadDof(localGridptr->template getEntitiesCount< Cell >());
        SharedPointer<MeshFunctionType> loadMeshFunctionptr;
        loadMeshFunctionptr->bind(loadGridptr,loadDof);

        loadDof.setValue(-1);
        
        File file;
        file.open( FileName+String("-")+distrgrid.printProcessCoords(), IOMode::read );
        loadMeshFunctionptr->boundLoad(file);
        file.close();

        for(int i=0;i<localDof.getSize();i++)
        {
            EXPECT_EQ( localDof.getElement(i), loadDof.getElement(i)) << "Compare Loaded and evaluated Dof Failed for: "<< i;
        }
    }
    
    static void TestLoad()
    {
        SharedPointer< LinearFunctionType, Device > linearFunctionPtr;
        MeshFunctionEvaluator< MeshFunctionType, LinearFunctionType > linearFunctionEvaluator;    
        
        ParameterProvider<dim,Device> parameters;

        //Crete distributed grid            
        PointType globalOrigin;
        globalOrigin.setValue(-0.5);

        PointType globalProportions;
        globalProportions.setValue(20);

        MeshType globalGrid;
        globalGrid.setDimensions(globalProportions);
        globalGrid.setDomain(globalOrigin,globalProportions);

        CoordinatesType overlap;
        overlap.setValue(1);
        DistributedGridType distrgrid;
        distrgrid.setDomainDecomposition( parameters.getDistr() );
        distrgrid.template setGlobalGrid<CommunicatorType>(globalGrid,overlap);

        //save files from local mesh        
        PointType localOrigin=parameters.getOrigin(CommunicatorType::GetRank());        
        PointType localProportions=parameters.getProportions(CommunicatorType::GetRank());;
            
        SharedPointer<MeshType> localGridptr;
        localGridptr->setDimensions(localProportions);
        localGridptr->setDomain(localOrigin,localProportions);

        DofType localDof(localGridptr->template getEntitiesCount< Cell >());

        SharedPointer<MeshFunctionType> localMeshFunctionptr;
        localMeshFunctionptr->bind(localGridptr,localDof);
        linearFunctionEvaluator.evaluateAllEntities(localMeshFunctionptr , linearFunctionPtr);


        String FileName=String("/tmp/test-file.tnl");
        File file;
        file.open( FileName+String("-")+distrgrid.printProcessCoords(), IOMode::write );        
        localMeshFunctionptr->save(file);
        file.close();



        //Crete "distributedgrid driven" grid filed by load
        SharedPointer<MeshType> loadGridptr;
        SharedPointer<MeshFunctionType> loadMeshFunctionptr;
        distrgrid.setupGrid(*loadGridptr);
        
        DofType loadDof(loadGridptr->template getEntitiesCount< Cell >());
        loadDof.setValue(0);
        loadMeshFunctionptr->bind(loadGridptr,loadDof);

        DistributedGridIO<MeshFunctionType> ::load(FileName, *loadMeshFunctionptr );

        loadMeshFunctionptr->template synchronize<CommunicatorType>(); //need synchronization for overlaps to be filled corectly in loadDof


        //Crete "distributedgrid driven" grid filed by evaluated linear function
        SharedPointer<MeshType> gridptr;
        SharedPointer<MeshFunctionType> meshFunctionptr;
        distrgrid.setupGrid(*gridptr);
        
        DofType dof(gridptr->template getEntitiesCount< Cell >());
        dof.setValue(-1);
        meshFunctionptr->bind(gridptr,dof);
        
        linearFunctionEvaluator.evaluateAllEntities(meshFunctionptr , linearFunctionPtr);        
        meshFunctionptr->template synchronize<CommunicatorType>();

        for(int i=0;i<dof.getSize();i++)
        {
            EXPECT_EQ( dof.getElement(i), loadDof.getElement(i)) << "Compare Loaded and evaluated Dof Failed for: "<< i;
        }       
    }
};

