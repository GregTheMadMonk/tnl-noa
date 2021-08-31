/***************************************************************************
                          tnl-benchmark-distributed-spmv.h  -  description
                             -------------------
    begin                : Sep 11, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovsky

#pragma once

#ifndef NDEBUG
#include <TNL/Debugging/FPE.h>
#endif

#include <TNL/Config/parseCommandLine.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/MPI/ScopedInitializer.h>
#include <TNL/MPI/Config.h>
#include <TNL/Containers/Partitioner.h>
#include <TNL/Containers/DistributedVector.h>
#include <TNL/Matrices/DistributedMatrix.h>
#include <TNL/Matrices/SparseOperations.h>

#include "../Benchmarks.h"
#include "ordering.h"

#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Algorithms/Segments/SlicedEllpack.h>

template< typename _Device, typename _Index, typename _IndexAlocator >
using SegmentsType = TNL::Algorithms::Segments::SlicedEllpack< _Device, _Index, _IndexAlocator >;

using namespace TNL;
using namespace TNL::Benchmarks;


template< typename Matrix, typename Vector >
void
benchmarkSpmv( Benchmark& benchmark,
               const Matrix& matrix,
               const Vector& x,
               const char* performer = "CPU" )
{
   Vector y;
   y.setLike( x );

   // reset function
   auto reset = [&]() {
      y = x;
   };

   // benchmark function
   auto compute = [&]() {
      matrix.vectorProduct( x, y );
   };

   benchmark.time< typename Matrix::DeviceType >( reset, performer, compute );
}

template< typename Matrix, typename Vector >
void
benchmarkSpmvCuda( Benchmark& benchmark,
                   const Matrix& matrix,
                   const Vector& x )
{
   using RealType = typename Matrix::RealType;
   using IndexType = typename Matrix::IndexType;
   using CudaMatrix = typename Matrix::template Self< RealType, Devices::Cuda >;
   using CudaVector = typename Vector::template Self< typename Vector::RealType, Devices::Cuda >;

   CudaVector cuda_x;
   cuda_x = x;

   Timer t;
   t.start();

   CudaMatrix cuda_matrix;
   cuda_matrix = matrix;

   t.stop();
   std::cout << "--> Copying the matrix to the GPU took " << t.getRealTime() << " seconds." << std::endl;

   benchmarkSpmv( benchmark, cuda_matrix, cuda_x, "GPU" );
}

template< typename Matrix, typename Vector >
void
benchmarkDistributedSpmv( Benchmark& benchmark,
                          // TODO: cannot be const due to internal buffering
//                          const Matrix& matrix,
                          Matrix& matrix,
                          const Vector& x,
                          const char* performer = "CPU" )
{
   Vector y;
   y.setLike( x );

   // reset function
   auto reset = [&]() {
      y = x;
   };

   // benchmark function
   auto compute = [&]() {
      matrix.vectorProduct( x, y );
      TNL::MPI::Barrier( matrix.getCommunicator() );
   };

   benchmark.time< typename Matrix::DeviceType >( reset, performer, compute );
}

template< typename Matrix, typename Vector >
void
benchmarkDistributedSpmvCuda( Benchmark& benchmark,
                              const Matrix& matrix,
                              const Vector& x )
{
   using RealType = typename Matrix::RealType;
   using IndexType = typename Matrix::IndexType;
   using CudaMatrix = typename Matrix::template Self< RealType, Devices::Cuda >;
   using CudaVector = typename Vector::template Self< typename Vector::RealType, Devices::Cuda >;

   CudaVector cuda_x;
   cuda_x = x;

   Timer t;
   t.start();

   CudaMatrix cuda_matrix;
   cuda_matrix = matrix;

   t.stop();
   std::cout << "--> Copying the matrix to the GPU took " << t.getRealTime() << " seconds." << std::endl;

   benchmarkDistributedSpmv( benchmark, cuda_matrix, cuda_x, "GPU" );
}

template< typename MatrixType >
struct SpmvBenchmark
{
   using RealType = typename MatrixType::RealType;
   using DeviceType = typename MatrixType::DeviceType;
   using IndexType = typename MatrixType::IndexType;
   using VectorType = Containers::Vector< RealType, DeviceType, IndexType >;

   using Partitioner = Containers::Partitioner< IndexType >;
   using DistributedMatrix = Matrices::DistributedMatrix< MatrixType >;
   using DistributedVector = Containers::DistributedVector< RealType, DeviceType, IndexType >;
   using DistributedRowLengths = typename DistributedMatrix::RowsCapacitiesType;

   static bool
   run( Benchmark& benchmark,
        Benchmark::MetadataMap metadata,
        const Config::ParameterContainer& parameters )
   {
      MatrixType matrix;
      VectorType vector;
      matrix.load( parameters.getParameter< String >( "input-matrix" ) );
      File( parameters.getParameter< String >( "input-vector" ), std::ios_base::in ) >> vector;

      typename MatrixType::RowsCapacitiesType rowLengths;
      matrix.getCompressedRowLengths( rowLengths );
      const IndexType maxRowLength = max( rowLengths );

      const String name = String( (TNL::MPI::GetSize() > 1) ? "DistSpMV" : "SpMV" )
                          + " (" + parameters.getParameter< String >( "name" ) + "): ";
      benchmark.newBenchmark( name, metadata );
      benchmark.setMetadataColumns( Benchmark::MetadataColumns({
         // TODO: strip the device
//         { "matrix type", matrix.getType() },
         { "rows", convertToString( matrix.getRows() ) },
         { "columns", convertToString( matrix.getColumns() ) },
         // FIXME: getMaxRowLengths() returns 0 for matrices loaded from file
//         { "max elements per row", matrix.getMaxRowLength() },
         { "max elements per row", convertToString( maxRowLength ) },
      } ));

      const bool reorder = parameters.getParameter< bool >( "reorder-dofs" );
      if( reorder ) {
         using PermutationVector = Containers::Vector< IndexType, DeviceType, IndexType >;
         PermutationVector perm, iperm;
         getTrivialOrdering( matrix, perm, iperm );
         MatrixType matrix_perm;
         Matrices::reorderSparseMatrix( matrix, matrix_perm, perm, iperm );
         if( TNL::MPI::GetSize() > 1 )
            runDistributed( benchmark, metadata, parameters, matrix_perm, vector );
         else
            runNonDistributed( benchmark, metadata, parameters, matrix_perm, vector );
      }
      else {
         if( TNL::MPI::GetSize() > 1 )
            runDistributed( benchmark, metadata, parameters, matrix, vector );
         else
            runNonDistributed( benchmark, metadata, parameters, matrix, vector );
      }

      return true;
   }

   static void
   runNonDistributed( Benchmark& benchmark,
                      Benchmark::MetadataMap metadata,
                      const Config::ParameterContainer& parameters,
                      MatrixType& matrix,
                      VectorType& vector )
   {
      benchmarkSpmv( benchmark, matrix, vector );
#ifdef HAVE_CUDA
      benchmarkSpmvCuda( benchmark, matrix, vector );
#endif
   }

   static void
   runDistributed( Benchmark& benchmark,
                   Benchmark::MetadataMap metadata,
                   const Config::ParameterContainer& parameters,
                   MatrixType& matrix,
                   VectorType& vector )
   {
      // set up the distributed matrix
      const auto communicator = MPI_COMM_WORLD;
      const auto localRange = Partitioner::splitRange( matrix.getRows(), communicator );
      DistributedMatrix distributedMatrix( localRange, matrix.getRows(), matrix.getColumns(), communicator );
      DistributedVector distributedVector( localRange, 0, matrix.getRows(), communicator );

      // copy the row lengths from the global matrix to the distributed matrix
      DistributedRowLengths distributedRowLengths( localRange, 0, matrix.getRows(), communicator );
      for( IndexType i = 0; i < distributedMatrix.getLocalMatrix().getRows(); i++ ) {
         const auto gi = distributedMatrix.getLocalRowRange().getGlobalIndex( i );
         distributedRowLengths[ gi ] = matrix.getRowCapacity( gi );
      }
      distributedMatrix.setRowCapacities( distributedRowLengths );

      // copy data from the global matrix/vector into the distributed matrix/vector
      for( IndexType i = 0; i < distributedMatrix.getLocalMatrix().getRows(); i++ ) {
         const auto gi = distributedMatrix.getLocalRowRange().getGlobalIndex( i );
         distributedVector[ gi ] = vector[ gi ];

//         const IndexType rowLength = matrix.getRowLength( i );
//         IndexType columns[ rowLength ];
//         RealType values[ rowLength ];
//         matrix.getRowFast( gi, columns, values );
//         distributedMatrix.setRowFast( gi, columns, values, rowLength );
         const auto global_row = matrix.getRow( gi );
         auto local_row = distributedMatrix.getRow( gi );
         for( IndexType j = 0; j < global_row.getSize(); j++ )
            local_row.setElement( j, global_row.getColumnIndex( j ), global_row.getValue( j ) );
      }

      benchmarkDistributedSpmv( benchmark, distributedMatrix, distributedVector );
#ifdef HAVE_CUDA
      benchmarkDistributedSpmvCuda( benchmark, distributedMatrix, distributedVector );
#endif

#ifndef NDEBUG
      // compare results of normal and distributed spmv
      VectorType y;
      y.setLike( vector );
      matrix.vectorProduct( vector, y );
      DistributedVector distributedY;
      distributedY.setLike( distributedVector );
      distributedMatrix.vectorProduct( distributedVector, distributedY );
      const int rank = TNL::MPI::GetRank( distributedMatrix.getCommunicator() );
      const int nproc = TNL::MPI::GetSize( distributedMatrix.getCommunicator() );
      typename VectorType::ViewType subY( &y[ Partitioner::getOffset( matrix.getRows(), rank, nproc ) ],
                                          Partitioner::getSizeForRank( matrix.getRows(), rank, nproc ) );
      TNL_ASSERT_EQ( distributedY.getLocalView(), subY, "WRONG RESULT !!!" );
#endif
   }
};

void
configSetup( Config::ConfigDescription & config )
{
   config.addDelimiter( "Benchmark settings:" );
   config.addEntry< String >( "log-file", "Log file name.", "tnl-benchmark-distributed-spmv.log");
   config.addEntry< String >( "output-mode", "Mode for opening the log file.", "overwrite" );
   config.addEntryEnum( "append" );
   config.addEntryEnum( "overwrite" );
   config.addEntry< int >( "loops", "Number of repetitions of the benchmark.", 10 );
   config.addRequiredEntry< String >( "input-matrix", "File name of the input matrix (in binary TNL format)." );
   config.addRequiredEntry< String >( "input-vector", "File name of the input vector (in binary TNL format)." );
   config.addEntry< String >( "name", "Name of the matrix in the benchmark.", "" );
   config.addEntry< int >( "verbose", "Verbose mode.", 1 );
   config.addEntry< bool >( "reorder-dofs", "Reorder matrix entries corresponding to the same DOF together.", false );

   config.addDelimiter( "Device settings:" );
   Devices::Host::configSetup( config );
   Devices::Cuda::configSetup( config );
   TNL::MPI::configSetup( config );
}

int
main( int argc, char* argv[] )
{
#ifndef NDEBUG
   Debugging::trackFloatingPointExceptions();
#endif

   Config::ParameterContainer parameters;
   Config::ConfigDescription conf_desc;

   configSetup( conf_desc );

   TNL::MPI::ScopedInitializer mpi(argc, argv);
   const int rank = TNL::MPI::GetRank();

   if( ! parseCommandLine( argc, argv, conf_desc, parameters ) )
      return EXIT_FAILURE;

   if( ! Devices::Host::setup( parameters ) ||
       ! Devices::Cuda::setup( parameters ) ||
       ! TNL::MPI::setup( parameters ) )
      return EXIT_FAILURE;

   const String & logFileName = parameters.getParameter< String >( "log-file" );
   const String & outputMode = parameters.getParameter< String >( "output-mode" );
   const int loops = parameters.getParameter< int >( "loops" );
   const int verbose = (rank == 0) ? parameters.getParameter< int >( "verbose" ) : 0;

   // open log file
   auto mode = std::ios::out;
   if( outputMode == "append" )
       mode |= std::ios::app;
   std::ofstream logFile;
   if( rank == 0 )
      logFile.open( logFileName.getString(), mode );

   // init benchmark and common metadata
   Benchmark benchmark( loops, verbose );

   // prepare global metadata
   Benchmark::MetadataMap metadata = getHardwareMetadata();

   // TODO: implement resolveMatrixType
//   return ! Matrices::resolveMatrixType< MainConfig,
//                                         Devices::Host,
//                                         SpmvBenchmark >( benchmark, metadata, parameters );
   using MatrixType = TNL::Matrices::SparseMatrix< double,
                                                   Devices::Host,
                                                   int,
                                                   TNL::Matrices::GeneralMatrix,
                                                   SegmentsType
                                                 >;
   const bool status = SpmvBenchmark< MatrixType >::run( benchmark, metadata, parameters );

   if( rank == 0 )
      if( ! benchmark.save( logFile ) ) {
         std::cerr << "Failed to write the benchmark results to file '" << parameters.getParameter< String >( "log-file" ) << "'." << std::endl;
         return EXIT_FAILURE;
      }

   return ! status;
}
