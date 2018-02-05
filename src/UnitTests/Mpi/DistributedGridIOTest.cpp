/***************************************************************************
                          CopyEntitiesTest.cpp  -  description
                             -------------------
    begin                : Nov 1, 2017
    copyright            : (C) 2017 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

#ifdef HAVE_GTEST  
#include <gtest/gtest.h>

#ifdef HAVE_MPI    

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
template <int dim>
class ParameterProvider
{
    public:

    typedef Grid<dim,double,Host,int> MeshType;
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

template<>
class ParameterProvider<1>
{
    public:

    int distr[1];

    typedef Grid<1,double,Host,int> MeshType;
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

    int* getDistr()
    {
        distr[0]=4;
        return distr;
    };
};

template<>
class ParameterProvider<2>
{
    public:

    int distr[2];

    typedef Grid<2,double,Host,int> MeshType;
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

    int* getDistr()
    {
        distr[0]=2;
        distr[1]=2;
        return distr;
    };
};

template<>
class ParameterProvider<3>
{
    public:

    int distr[3];

    typedef Grid<3,double,Host,int> MeshType;
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

    int* getDistr()
    {
        distr[0]=2;
        distr[1]=2;
        distr[2]=1;
        return distr;
    };
};

//------------------------------------------------------------------------------

typedef MpiCommunicator CommunicatorType;

template <int dim>
class TestDistributedGridIO{
    public:

    typedef Grid<dim,double,Host,int> MeshType;
    typedef MeshFunction<MeshType> MeshFunctionType;
    typedef Vector<double,Host,int> DofType;
    typedef typename MeshType::Cell Cell;
    typedef typename MeshType::IndexType IndexType; 
    typedef typename MeshType::PointType PointType; 
    typedef DistributedMesh<MeshType> DistributedGridType;

    typedef typename DistributedGridType::CoordinatesType CoordinatesType;
    typedef LinearFunction<double,dim> LinearFunctionType;

    static void TestSave()
    {
        SharedPointer< LinearFunctionType, Host > linearFunctionPtr;
        MeshFunctionEvaluator< MeshFunctionType, LinearFunctionType > linearFunctionEvaluator;    
        
        ParameterProvider<dim> parametry;
        
        //save distributed meshfunction into files
        PointType globalOrigin;
        globalOrigin.setValue(-0.5);

        PointType globalProportions;
        globalProportions.setValue(20);


        MeshType globalGrid;
        globalGrid.setDimensions(globalProportions);
        globalGrid.setDomain(globalOrigin,globalProportions);
        
        int *distr=parametry.getDistr();

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
        
        File file;

        File meshFile;
        file.open( String( "/tmp/test-file.tnl-" )+convertToString(CommunicatorType::GetRank()), IOMode::write );
        meshFile.open( String( "/tmp/test-file-mesh.tnl-" )+convertToString(CommunicatorType::GetRank()), IOMode::write );
        DistributedGridIO<MeshFunctionType> ::save(file,meshFile, *meshFunctionptr );
        meshFile.close();

        file.close();

        //create similar local mesh function and evaluate linear function on it
        PointType localOrigin=parametry.getOrigin(CommunicatorType::GetRank());        
        PointType localProportions=parametry.getProportions(CommunicatorType::GetRank());;
            
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
        

        file.open( String( "/tmp/test-file.tnl-" )+convertToString(CommunicatorType::GetRank()), IOMode::read );
        loadMeshFunctionptr->boundLoad(file);
        file.close();

        for(int i=0;i<localDof.getSize();i++)
        {
            EXPECT_EQ( localDof[i], loadDof[i]) << "Compare Loaded and evaluated Dof Failed for: "<< i;
        }
        
    }
    
    static void TestLoad()
    {
        SharedPointer< LinearFunctionType, Host > linearFunctionPtr;
        MeshFunctionEvaluator< MeshFunctionType, LinearFunctionType > linearFunctionEvaluator;    
        
        ParameterProvider<dim> parametry;

        //save files from local mesh        
        PointType localOrigin=parametry.getOrigin(CommunicatorType::GetRank());        
        PointType localProportions=parametry.getProportions(CommunicatorType::GetRank());;
            
        SharedPointer<MeshType> localGridptr;
        localGridptr->setDimensions(localProportions);
        localGridptr->setDomain(localOrigin,localProportions);

        DofType localDof(localGridptr->template getEntitiesCount< Cell >());

        SharedPointer<MeshFunctionType> localMeshFunctionptr;
        localMeshFunctionptr->bind(localGridptr,localDof);
        linearFunctionEvaluator.evaluateAllEntities(localMeshFunctionptr , linearFunctionPtr);

        File file;
        file.open( String( "/tmp/test-file.tnl-" )+convertToString(CommunicatorType::GetRank()), IOMode::write );        
        localMeshFunctionptr->save(file);
        file.close();

        //Crete distributed grid            
        PointType globalOrigin;
        globalOrigin.setValue(-0.5);

        PointType globalProportions;
        globalProportions.setValue(20);

        MeshType globalGrid;
        globalGrid.setDimensions(globalProportions);
        globalGrid.setDomain(globalOrigin,globalProportions);
        
        int *distr=parametry.getDistr();

        CoordinatesType overlap;
        overlap.setValue(1);
        DistributedGridType distrgrid;
        distrgrid.template setGlobalGrid<CommunicatorType>(globalGrid,overlap, distr);

        //Crete "distributedgrid driven" grid filed by load
        SharedPointer<MeshType> loadGridptr;
        SharedPointer<MeshFunctionType> loadMeshFunctionptr;
        distrgrid.SetupGrid(*loadGridptr);
        
        DofType loadDof(loadGridptr->template getEntitiesCount< Cell >());
        loadDof.setValue(0);
        loadMeshFunctionptr->bind(loadGridptr,loadDof);

            
        file.open( String( "/tmp/test-file.tnl-" )+convertToString(CommunicatorType::GetRank()), IOMode::read );    
        DistributedGridIO<MeshFunctionType> ::load(file, *loadMeshFunctionptr );
        file.close();

        loadMeshFunctionptr->template Synchronize<CommunicatorType>(); //need synchronization for overlaps to be filled corectly in loadDof


        //Crete "distributedgrid driven" grid filed by evaluated linear function
        SharedPointer<MeshType> gridptr;
        SharedPointer<MeshFunctionType> meshFunctionptr;
        distrgrid.SetupGrid(*gridptr);
        
        DofType dof(gridptr->template getEntitiesCount< Cell >());
        dof.setValue(-1);
        meshFunctionptr->bind(gridptr,dof);
        
        linearFunctionEvaluator.evaluateAllEntities(meshFunctionptr , linearFunctionPtr);        
        meshFunctionptr->template Synchronize<CommunicatorType>();

        for(int i=0;i<localDof.getSize();i++)
        {
            EXPECT_EQ( dof[i], loadDof[i]) << "Compare Loaded and evaluated Dof Failed for: "<< i;
        }        
    }
};


TEST( DistributedGridIO, Save_1D )
{
    TestDistributedGridIO<1>::TestSave();
}

TEST( DistributedGridIO, Save_2D )
{
    TestDistributedGridIO<2>::TestSave();
}

TEST( DistributedGridIO, Save_3D )
{
    TestDistributedGridIO<3>::TestSave();
}

TEST( DistributedGridIO, Load_1D )
{
    TestDistributedGridIO<1>::TestLoad();
}

TEST( DistributedGridIO, Load_2D )
{
    TestDistributedGridIO<2>::TestLoad();
}

TEST( DistributedGridIO, Load_3D )
{
    TestDistributedGridIO<3>::TestLoad();
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

  class MinimalistBuffredPrinter : public ::testing::EmptyTestEventListener {
      
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
        int rank=CommunicatorType::GetRank();
        sout<< test_info.test_case_name() <<"." << test_info.name() << " End." <<std::endl;
        std::cout << rank << ":" << std::endl << sout.str()<< std::endl;
        sout.str( std::string() );
        sout.clear();
    }
  };
#endif

#include "../GtestMissingError.h"
int main( int argc, char* argv[] )
{
#ifdef HAVE_GTEST
   ::testing::InitGoogleTest( &argc, argv );

    #ifdef HAVE_MPI
       ::testing::TestEventListeners& listeners =
          ::testing::UnitTest::GetInstance()->listeners();

       delete listeners.Release(listeners.default_result_printer());
       listeners.Append(new MinimalistBuffredPrinter);

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
