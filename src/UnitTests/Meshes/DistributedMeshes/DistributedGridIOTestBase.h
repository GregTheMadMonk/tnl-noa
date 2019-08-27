/***************************************************************************
                          DistributedGridIO.h  -  description
                             -------------------
    begin                : Nov 1, 2017
    copyright            : (C) 2017 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/
#ifdef HAVE_GTEST
  
#include <gtest/gtest.h>

#ifdef HAVE_MPI

#include "DistributedGridIOTest.h"

TEST( DistributedGridIO, Save_1D )
{
    TestDistributedGridIO<1,Devices::Host>::TestSave();
}

TEST( DistributedGridIO, Save_2D )
{
    TestDistributedGridIO<2,Devices::Host>::TestSave();
}

TEST( DistributedGridIO, Save_3D )
{
    TestDistributedGridIO<3,Devices::Host>::TestSave();
}

TEST( DistributedGridIO, Load_1D )
{
    TestDistributedGridIO<1,Devices::Host>::TestLoad();
}

TEST( DistributedGridIO, Load_2D )
{
    TestDistributedGridIO<2,Devices::Host>::TestLoad();
}

TEST( DistributedGridIO, Load_3D )
{
    TestDistributedGridIO<3,Devices::Host>::TestLoad();
}

#ifdef HAVE_CUDA
TEST( DistributedGridIO, Save_1D_GPU )
{
    TestDistributedGridIO<1,Devices::Cuda>::TestSave();
}

TEST( DistributedGridIO, Save_2D_GPU )
{
    TestDistributedGridIO<2,Devices::Cuda>::TestSave();
}

TEST( DistributedGridIO, Save_3D_GPU )
{
    TestDistributedGridIO<3,Devices::Cuda>::TestSave();
}

TEST( DistributedGridIO, Load_1D_GPU )
{
    TestDistributedGridIO<1,Devices::Cuda>::TestLoad();
}

TEST( DistributedGridIO, Load_2D_GPU )
{
    TestDistributedGridIO<2,Devices::Cuda>::TestLoad();
}

TEST( DistributedGridIO, Load_3D_GPU )
{
    TestDistributedGridIO<3,Devices::Cuda>::TestLoad();
}
#endif

#endif

#endif

#include "../../main_mpi.h"
