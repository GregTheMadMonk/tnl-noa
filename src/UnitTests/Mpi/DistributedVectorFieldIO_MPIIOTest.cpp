


#ifdef HAVE_GTEST
      #include <gtest/gtest.h>
#ifdef HAVE_MPI

#include <TNL/Communicators/MpiCommunicator.h>
#include "DistributedVectorFieldIO_MPIIOTestBase.h"

using namespace TNL::Communicators;

typedef MpiCommunicator CommunicatorType;

TEST( DistributedVectorFieldIO_MPIIO, Save_1D )
{
    TestDistributedVectorFieldMPIIO<1,2,Host>::TestSave();
}

TEST( DistributedVectorFieldIO_MPIIO, Save_2D )
{
    TestDistributedVectorFieldMPIIO<2,3,Host>::TestSave();
}

TEST( DistributedVectorFieldIO_MPIIO, Save_3D )
{
    TestDistributedVectorFieldMPIIO<3,2,Host>::TestSave();
}


TEST( DistributedVectorFieldIO_MPIIO, Load_1D )
{
    TestDistributedVectorFieldMPIIO<1,2,Host>::TestLoad();
}

TEST( DistributedVectorFieldIO_MPIIO, Load_2D )
{
    TestDistributedVectorFieldMPIIO<2,3,Host>::TestLoad();
}

TEST( DistributedVectorFieldIO_MPIIO, Load_3D )
{
    TestDistributedVectorFieldMPIIO<3,2,Host>::TestLoad();
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

  class MinimalistBufferedPrinter : public ::testing::EmptyTestEventListener {
      
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
        int rank=CommunicatorType::GetRank(CommunicatorType::AllGroup);
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
       listeners.Append(new MinimalistBufferedPrinter);

       CommunicatorType::Init(argc,argv );
       CommunicatorType::setRedirection( false );
       CommunicatorType::setupRedirection();
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
