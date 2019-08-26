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

TEST( DistributedGridMPIIO, Save_1D )
{
    TestDistributedGridMPIIO<1,Devices::Host>::TestSave();
}

TEST( DistributedGridMPIIO, Save_2D )
{
    TestDistributedGridMPIIO<2,Devices::Host>::TestSave();
}

TEST( DistributedGridMPIIO, Save_3D )
{
    TestDistributedGridMPIIO<3,Devices::Host>::TestSave();
}

TEST( DistributedGridMPIIO, Load_1D )
{
    TestDistributedGridMPIIO<1,Devices::Host>::TestLoad();
}

TEST( DistributedGridMPIIO, Load_2D )
{
    TestDistributedGridMPIIO<2,Devices::Host>::TestLoad();
}

TEST( DistributedGridMPIIO, Load_3D )
{
    TestDistributedGridMPIIO<3,Devices::Host>::TestLoad();
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

#endif

#endif

#include "../../main_mpi.h"
