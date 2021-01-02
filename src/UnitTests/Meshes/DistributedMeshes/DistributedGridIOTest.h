/***************************************************************************
                          DistributedGridIO.h  -  description
                             -------------------
    begin                : Nov 1, 2017
    copyright            : (C) 2017 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

#include <TNL/Meshes/DistributedMeshes/DistributedMesh.h>
#include <TNL/Functions/MeshFunctionView.h>
#include <TNL/Meshes/DistributedMeshes/DistributedMeshSynchronizer.h>
#include <TNL/Meshes/DistributedMeshes/DistributedGridIO.h>
#include <TNL/Meshes/DistributedMeshes/SubdomainOverlapsGetter.h>

#include "../../Functions/Functions.h"

using namespace TNL::Containers;
using namespace TNL::Meshes;
using namespace TNL::Functions;
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

template <int dim, typename Device>
class TestDistributedGridIO
{
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
        DistributedGridType distributedGrid;
        distributedGrid.setDomainDecomposition( parameters.getDistr() );
        distributedGrid.setGlobalGrid( globalGrid );
        typename DistributedGridType::SubdomainOverlapsType lowerOverlap, upperOverlap;
        SubdomainOverlapsGetter< MeshType >::getOverlaps( &distributedGrid, lowerOverlap, upperOverlap, 1 );
        distributedGrid.setOverlaps( lowerOverlap, upperOverlap );

        //std::cout << distributedGrid.printProcessDistr() <<std::endl;

        Pointers::SharedPointer<MeshType> gridptr;
        Pointers::SharedPointer<MeshFunctionType> meshFunctionptr;
        distributedGrid.setupGrid(*gridptr);

        DofType dof(gridptr->template getEntitiesCount< Cell >());
        dof.setValue(0);
        meshFunctionptr->bind(gridptr,dof);

        linearFunctionEvaluator.evaluateAllEntities(meshFunctionptr , linearFunctionPtr);

        String fileName=String("test-file-distriburtegrid-io-save.tnl");
        DistributedGridIO<MeshFunctionType> ::save(fileName, *meshFunctionptr );


       //create similar local mesh function and evaluate linear function on it
        PointType localOrigin=parameters.getOrigin(TNL::MPI::GetRank());
        PointType localProportions=parameters.getProportions(TNL::MPI::GetRank());

        Pointers::SharedPointer<MeshType>  localGridptr;
        localGridptr->setDimensions(localProportions);
        localGridptr->setDomain(localOrigin,localProportions);

        DofType localDof(localGridptr->template getEntitiesCount< Cell >());

        Pointers::SharedPointer<MeshFunctionType> localMeshFunctionptr;
        localMeshFunctionptr->bind(localGridptr,localDof);
        linearFunctionEvaluator.evaluateAllEntities(localMeshFunctionptr , linearFunctionPtr);

        //load other meshfunction on same localgrid from created file
        Pointers::SharedPointer<MeshType>  loadGridptr;
        loadGridptr->setDimensions(localProportions);
        loadGridptr->setDomain(localOrigin,localProportions);

        DofType loadDof(localGridptr->template getEntitiesCount< Cell >());
        Pointers::SharedPointer<MeshFunctionType> loadMeshFunctionptr;
        loadMeshFunctionptr->bind(loadGridptr,loadDof);

        loadDof.setValue(-1);

        String localFileName= fileName+String("-")+distributedGrid.printProcessCoords()+String(".tnl");

        File file;
        file.open(localFileName, std::ios_base::in );
        loadMeshFunctionptr->boundLoad(file);
        file.close();

        for(int i=0;i<localDof.getSize();i++)
        {
            EXPECT_EQ( localDof.getElement(i), loadDof.getElement(i)) << "Compare Loaded and evaluated Dof Failed for: "<< i;
        }

        EXPECT_EQ( std::remove( localFileName.getString()) , 0 );

       //remove meshfile
       EXPECT_EQ( std::remove( (fileName+String("-mesh-")+distributedGrid.printProcessCoords()+String(".tnl")).getString()) , 0 );
    }

    static void TestLoad()
    {
        Pointers::SharedPointer< LinearFunctionType, Device > linearFunctionPtr;
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
        DistributedGridType distributedGrid;
        distributedGrid.setDomainDecomposition( parameters.getDistr() );
        distributedGrid.setGlobalGrid( globalGrid );
        typename DistributedGridType::SubdomainOverlapsType lowerOverlap, upperOverlap;
        SubdomainOverlapsGetter< MeshType >::getOverlaps( &distributedGrid, lowerOverlap, upperOverlap, 1 );
        distributedGrid.setOverlaps( lowerOverlap, upperOverlap );

        //save files from local mesh
        PointType localOrigin=parameters.getOrigin(TNL::MPI::GetRank());
        PointType localProportions=parameters.getProportions(TNL::MPI::GetRank());

        Pointers::SharedPointer<MeshType> localGridptr;
        localGridptr->setDimensions(localProportions);
        localGridptr->setDomain(localOrigin,localProportions);

        DofType localDof(localGridptr->template getEntitiesCount< Cell >());

        Pointers::SharedPointer<MeshFunctionType> localMeshFunctionptr;
        localMeshFunctionptr->bind(localGridptr,localDof);
        linearFunctionEvaluator.evaluateAllEntities(localMeshFunctionptr , linearFunctionPtr);


        String fileName=String("test-file-distributedgrid-io-load.tnl");
        String localFileName=fileName+String("-")+distributedGrid.printProcessCoords()+String(".tnl");
        File file;
        file.open( localFileName, std::ios_base::out );
        localMeshFunctionptr->save(file);
        file.close();



        //Crete "distributedgrid driven" grid filed by load
        Pointers::SharedPointer<MeshType> loadGridptr;
        Pointers::SharedPointer<MeshFunctionType> loadMeshFunctionptr;
        distributedGrid.setupGrid(*loadGridptr);

        DofType loadDof(loadGridptr->template getEntitiesCount< Cell >());
        loadDof.setValue(0);
        loadMeshFunctionptr->bind(loadGridptr,loadDof);

        DistributedGridIO<MeshFunctionType> ::load(fileName, *loadMeshFunctionptr );

        DistributedMeshSynchronizer< DistributedGridType > synchronizer;
        synchronizer.setDistributedGrid( &distributedGrid );
        synchronizer.synchronize( *loadMeshFunctionptr ); //need synchronization for overlaps to be filled corectly in loadDof

        //Crete "distributedgrid driven" grid filed by evaluated linear function
        Pointers::SharedPointer<MeshType> gridptr;
        Pointers::SharedPointer<MeshFunctionType> meshFunctionptr;
        distributedGrid.setupGrid(*gridptr);

        DofType dof(gridptr->template getEntitiesCount< Cell >());
        dof.setValue(-1);
        meshFunctionptr->bind(gridptr,dof);

        linearFunctionEvaluator.evaluateAllEntities(meshFunctionptr , linearFunctionPtr);
        synchronizer.synchronize( *meshFunctionptr );

        for(int i=0;i<dof.getSize();i++)
        {
            EXPECT_EQ( dof.getElement(i), loadDof.getElement(i)) << "Compare Loaded and evaluated Dof Failed for: "<< i;
        }

        EXPECT_EQ( std::remove( localFileName.getString()) , 0 );
    }
};

