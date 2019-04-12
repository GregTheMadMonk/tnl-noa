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
#endif

#endif

#include "../../main_mpi.h"
