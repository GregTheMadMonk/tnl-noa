/***************************************************************************
                          SparseMatrixTest_SlicedEllpack.h -  description
                             -------------------
    begin                : Dec 3, 2019
    copyright            : (C) 2019 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <iostream>
#include <TNL/Containers/Segments/SlicedEllpack.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Matrices/MatrixType.h>

#ifdef HAVE_GTEST
#include <gtest/gtest.h>

const char* saveAndLoadFileName = "test_SparseMatrixTest_SlicedEllpack_segments";

////
// Row-major format is used for the host system
template< typename Device, typename Index, typename IndexAllocator >
using RowMajorSlicedEllpack = TNL::Containers::Segments::SlicedEllpack< Device, Index, IndexAllocator, true, 32 >;

////
// Column-major format is used for GPUs
template< typename Device, typename Index, typename IndexAllocator >
using ColumnMajorSlicedEllpack = TNL::Containers::Segments::SlicedEllpack< Device, Index, IndexAllocator, false, 32 >;

// types for which MatrixTest is instantiated
using SlicedEllpackMatrixTypes = ::testing::Types
<
    TNL::Matrices::SparseMatrix< int,     TNL::Devices::Host, int,   TNL::Matrices::GeneralMatrix, RowMajorSlicedEllpack >,
    TNL::Matrices::SparseMatrix< long,    TNL::Devices::Host, int,   TNL::Matrices::GeneralMatrix, RowMajorSlicedEllpack >,
    TNL::Matrices::SparseMatrix< float,   TNL::Devices::Host, int,   TNL::Matrices::GeneralMatrix, RowMajorSlicedEllpack >,
    TNL::Matrices::SparseMatrix< double,  TNL::Devices::Host, int,   TNL::Matrices::GeneralMatrix, RowMajorSlicedEllpack >,
    TNL::Matrices::SparseMatrix< int,     TNL::Devices::Host, long,  TNL::Matrices::GeneralMatrix, RowMajorSlicedEllpack >,
    TNL::Matrices::SparseMatrix< long,    TNL::Devices::Host, long,  TNL::Matrices::GeneralMatrix, RowMajorSlicedEllpack >,
    TNL::Matrices::SparseMatrix< float,   TNL::Devices::Host, long,  TNL::Matrices::GeneralMatrix, RowMajorSlicedEllpack >,
    TNL::Matrices::SparseMatrix< double,  TNL::Devices::Host, long,  TNL::Matrices::GeneralMatrix, RowMajorSlicedEllpack >
#ifdef HAVE_CUDA
   ,TNL::Matrices::SparseMatrix< int,     TNL::Devices::Cuda, int,   TNL::Matrices::GeneralMatrix, ColumnMajorSlicedEllpack >,
    TNL::Matrices::SparseMatrix< long,    TNL::Devices::Cuda, int,   TNL::Matrices::GeneralMatrix, ColumnMajorSlicedEllpack >,
    TNL::Matrices::SparseMatrix< float,   TNL::Devices::Cuda, int,   TNL::Matrices::GeneralMatrix, ColumnMajorSlicedEllpack >,
    TNL::Matrices::SparseMatrix< double,  TNL::Devices::Cuda, int,   TNL::Matrices::GeneralMatrix, ColumnMajorSlicedEllpack >,
    TNL::Matrices::SparseMatrix< int,     TNL::Devices::Cuda, long,  TNL::Matrices::GeneralMatrix, ColumnMajorSlicedEllpack >,
    TNL::Matrices::SparseMatrix< long,    TNL::Devices::Cuda, long,  TNL::Matrices::GeneralMatrix, ColumnMajorSlicedEllpack >,
    TNL::Matrices::SparseMatrix< float,   TNL::Devices::Cuda, long,  TNL::Matrices::GeneralMatrix, ColumnMajorSlicedEllpack >,
    TNL::Matrices::SparseMatrix< double,  TNL::Devices::Cuda, long,  TNL::Matrices::GeneralMatrix, ColumnMajorSlicedEllpack >
#endif
>;

#endif

#include "../main.h"
