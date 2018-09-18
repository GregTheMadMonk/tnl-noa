/***************************************************************************
                          DistributedGridIO_MPIIO  -  description
                             -------------------
    begin                : Nov 1, 2017
    copyright            : (C) 2017 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/
#ifdef HAVE_GTEST
      #include <gtest/gtest.h>
#ifdef HAVE_MPI

#include "DistributedGridIO_MPIIOTest.h"
#include <TNL/Communicators/ScopedInitializer.h>

TEST( DistributedGridMPIIO, Save_1D )
{
    TestDistributedGridMPIIO<1,Host>::TestSave();
}

TEST( DistributedGridMPIIO, Save_2D )
{
    TestDistributedGridMPIIO<2,Host>::TestSave();
}

TEST( DistributedGridMPIIO, Save_3D )
{
    TestDistributedGridMPIIO<3,Host>::TestSave();
}

TEST( DistributedGridMPIIO, Load_1D )
{
    TestDistributedGridMPIIO<1,Host>::TestLoad();
}

TEST( DistributedGridMPIIO, Load_2D )
{
    TestDistributedGridMPIIO<2,Host>::TestLoad();
}

TEST( DistributedGridMPIIO, Load_3D )
{
    TestDistributedGridMPIIO<3,Host>::TestLoad();
}

#ifdef HAVE_CUDA
    TEST( DistributedGridMPIIO, Save_1D_GPU )
    {
        TestDistributedGridMPIIO<1,Cuda>::TestSave();
    }

    TEST( DistributedGridMPIIO, Save_2D_GPU )
    {
        TestDistributedGridMPIIO<2,Cuda>::TestSave();
    }

    TEST( DistributedGridMPIIO, Save_3D_GPU )
    {
        TestDistributedGridMPIIO<3,Cuda>::TestSave();
    }

    TEST( DistributedGridMPIIO, Load_1D_GPU )
    {
        TestDistributedGridMPIIO<1,Cuda>::TestLoad();
    }

    TEST( DistributedGridMPIIO, Load_2D_GPU )
    {
        TestDistributedGridMPIIO<2,Cuda>::TestLoad();
    }

    TEST( DistributedGridMPIIO, Load_3D_GPU )
    {
        TestDistributedGridMPIIO<3,Cuda>::TestLoad();
    }
#endif

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

#include "../../GtestMissingError.h"
int main( int argc, char* argv[] )
{
#ifdef HAVE_GTEST
   ::testing::InitGoogleTest( &argc, argv );

    #ifdef HAVE_MPI
       ::testing::TestEventListeners& listeners =
          ::testing::UnitTest::GetInstance()->listeners();

       delete listeners.Release(listeners.default_result_printer());
       listeners.Append(new MinimalistBufferedPrinter);

       Communicators::ScopedInitializer< CommunicatorType > mpi(argc, argv);
       CommunicatorType::setRedirection( false );
       CommunicatorType::setupRedirection();
    #endif
       return RUN_ALL_TESTS();
#else
   
   throw GtestMissingError();
#endif
}
