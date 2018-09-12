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

#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Devices/SystemInfo.h>
#include <TNL/Devices/CudaDeviceInfo.h>
#include <TNL/Config/ConfigDescription.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Communicators/MpiCommunicator.h>
#include <TNL/Communicators/NoDistrCommunicator.h>
#include <TNL/Communicators/ScopedInitializer.h>
#include <TNL/DistributedContainers/Partitioner.h>
#include <TNL/DistributedContainers/DistributedVector.h>
#include <TNL/DistributedContainers/DistributedMatrix.h>

#include "../Benchmarks.h"
#include "ordering.h"

#include <TNL/Matrices/SlicedEllpack.h>

using namespace TNL;
using namespace TNL::Benchmarks;

#ifdef HAVE_MPI
using CommunicatorType = Communicators::MpiCommunicator;
#else
using CommunicatorType = Communicators::NoDistrCommunicator;
#endif


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

   benchmark.time( reset, performer, compute );
}

template< typename Matrix, typename Vector >
void
benchmarkSpmvCuda( Benchmark& benchmark,
                   const Matrix& matrix,
                   const Vector& x )
{
   using RealType = typename Matrix::RealType;
   using IndexType = typename Matrix::IndexType;
   using CudaMatrix = typename Matrix::CudaType;
   using CudaVector = typename Vector::CudaType;

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
      Matrix::CommunicatorType::Barrier( matrix.getCommunicationGroup() );
   };

   benchmark.time( reset, performer, compute );
}

template< typename Matrix, typename Vector >
void
benchmarkDistributedSpmvCuda( Benchmark& benchmark,
                              const Matrix& matrix,
                              const Vector& x )
{
   using RealType = typename Matrix::RealType;
   using IndexType = typename Matrix::IndexType;
   using CudaMatrix = typename Matrix::CudaType;
   using CudaVector = typename Vector::CudaType;

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

   using Partitioner = DistributedContainers::Partitioner< IndexType, CommunicatorType >;
   using DistributedMatrix = DistributedContainers::DistributedMatrix< MatrixType, CommunicatorType >;
   using DistributedVector = DistributedContainers::DistributedVector< RealType, DeviceType, IndexType, CommunicatorType >;
   using DistributedRowLengths = typename DistributedMatrix::CompressedRowLengthsVector;

   static bool
   run( Benchmark& benchmark,
        Benchmark::MetadataMap metadata,
        const Config::ParameterContainer& parameters )
   {
      MatrixType matrix;
      VectorType vector;
      if( ! matrix.load( parameters.getParameter< String >( "input-matrix" ) ) ||
          ! vector.load( parameters.getParameter< String >( "input-vector" ) ) )
          return false;

      typename MatrixType::CompressedRowLengthsVector rowLengths;
      matrix.getCompressedRowLengths( rowLengths );
      const IndexType maxRowLength = rowLengths.max();

      const String name = String( (CommunicatorType::isDistributed()) ? "DistSpMV" : "SpMV" )
                          + " (" + parameters.getParameter< String >( "name" ) + "): ";
      benchmark.newBenchmark( name, metadata );
      benchmark.setMetadataColumns( Benchmark::MetadataColumns({
         // TODO: strip the device
//         {"matrix type", matrix.getType()},
         {"rows", matrix.getRows()},
         {"columns", matrix.getColumns()},
         // FIXME: getMaxRowLengths() returns 0 for matrices loaded from file
//         {"max elements per row", matrix.getMaxRowLength()},
         {"max elements per row", maxRowLength},
      } ));

      const bool reorder = parameters.getParameter< bool >( "reorder-dofs" );
      if( reorder ) {
         using PermutationVector = Containers::Vector< IndexType, DeviceType, IndexType >;
         PermutationVector perm, iperm;
         getTrivialOrdering( matrix, perm, iperm );
         MatrixType matrix_perm;
         reorderMatrix( matrix, matrix_perm, perm, iperm );
         if( CommunicatorType::isDistributed() )
            runDistributed( benchmark, metadata, parameters, matrix_perm, vector );
         else
            runNonDistributed( benchmark, metadata, parameters, matrix_perm, vector );
      }
      else {
         if( CommunicatorType::isDistributed() )
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
      const auto group = CommunicatorType::AllGroup;
      const auto localRange = Partitioner::splitRange( matrix.getRows(), group );
      DistributedMatrix distributedMatrix( localRange, matrix.getRows(), matrix.getColumns(), group );
      DistributedVector distributedVector( localRange, matrix.getRows(), group );

      // copy the row lengths from the global matrix to the distributed matrix
      DistributedRowLengths distributedRowLengths( localRange, matrix.getRows(), group );
      for( IndexType i = 0; i < distributedMatrix.getLocalMatrix().getRows(); i++ ) {
         const auto gi = distributedMatrix.getLocalRowRange().getGlobalIndex( i );
         distributedRowLengths[ gi ] = matrix.getRowLength( gi );
      }
      distributedMatrix.setCompressedRowLengths( distributedRowLengths );

      // copy data from the global matrix/vector into the distributed matrix/vector
      for( IndexType i = 0; i < distributedMatrix.getLocalMatrix().getRows(); i++ ) {
         const auto gi = distributedMatrix.getLocalRowRange().getGlobalIndex( i );
         distributedVector[ gi ] = vector[ gi ];

         const IndexType rowLength = matrix.getRowLength( i );
         IndexType columns[ rowLength ];
         RealType values[ rowLength ];
         matrix.getRowFast( gi, columns, values );
         distributedMatrix.setRowFast( gi, columns, values, rowLength );
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
      const int rank = CommunicatorType::GetRank( distributedMatrix.getCommunicationGroup() );
      const int nproc = CommunicatorType::GetSize( distributedMatrix.getCommunicationGroup() );
      VectorType subY( y,
                       Partitioner::getOffset( matrix.getRows(), rank, nproc ),
                       Partitioner::getSizeForRank( matrix.getRows(), rank, nproc ) );
      TNL_ASSERT_EQ( distributedY.getLocalVectorView(), subY, "WRONG RESULT !!!" );
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
   CommunicatorType::configSetup( config );
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

   Communicators::ScopedInitializer< CommunicatorType > scopedInit(argc, argv);
   const int rank = CommunicatorType::GetRank( CommunicatorType::AllGroup );

   if( ! parseCommandLine( argc, argv, conf_desc, parameters ) ) {
      conf_desc.printUsage( argv[ 0 ] );
      return EXIT_FAILURE;
   }

   if( ! Devices::Host::setup( parameters ) ||
       ! Devices::Cuda::setup( parameters ) ||
       ! CommunicatorType::setup( parameters ) )
      return EXIT_FAILURE;

   const String & logFileName = parameters.getParameter< String >( "log-file" );
   const String & outputMode = parameters.getParameter< String >( "output-mode" );
   const unsigned loops = parameters.getParameter< unsigned >( "loops" );
   const unsigned verbose = (rank == 0) ? parameters.getParameter< unsigned >( "verbose" ) : 0;

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
   const int cpu_id = 0;
   Devices::CacheSizes cacheSizes = Devices::SystemInfo::getCPUCacheSizes( cpu_id );
   String cacheInfo = String( cacheSizes.L1data ) + ", "
                       + String( cacheSizes.L1instruction ) + ", "
                       + String( cacheSizes.L2 ) + ", "
                       + String( cacheSizes.L3 );
#ifdef HAVE_CUDA
   const int activeGPU = Devices::CudaDeviceInfo::getActiveDevice();
   const String deviceArch = String( Devices::CudaDeviceInfo::getArchitectureMajor( activeGPU ) ) + "." +
                             String( Devices::CudaDeviceInfo::getArchitectureMinor( activeGPU ) );
#endif
   Benchmark::MetadataMap metadata {
       { "host name", Devices::SystemInfo::getHostname() },
       { "architecture", Devices::SystemInfo::getArchitecture() },
       { "system", Devices::SystemInfo::getSystemName() },
       { "system release", Devices::SystemInfo::getSystemRelease() },
       { "start time", Devices::SystemInfo::getCurrentTime() },
#ifdef HAVE_MPI
       { "number of MPI processes", CommunicatorType::GetSize( CommunicatorType::AllGroup ) },
#endif
       { "OpenMP enabled", Devices::Host::isOMPEnabled() },
       { "OpenMP threads", Devices::Host::getMaxThreadsCount() },
       { "CPU model name", Devices::SystemInfo::getCPUModelName( cpu_id ) },
       { "CPU cores", Devices::SystemInfo::getNumberOfCores( cpu_id ) },
       { "CPU threads per core", Devices::SystemInfo::getNumberOfThreads( cpu_id ) / Devices::SystemInfo::getNumberOfCores( cpu_id ) },
       { "CPU max frequency (MHz)", Devices::SystemInfo::getCPUMaxFrequency( cpu_id ) / 1e3 },
       { "CPU cache sizes (L1d, L1i, L2, L3) (kiB)", cacheInfo },
#ifdef HAVE_CUDA
       { "GPU name", Devices::CudaDeviceInfo::getDeviceName( activeGPU ) },
       { "GPU architecture", deviceArch },
       { "GPU CUDA cores", Devices::CudaDeviceInfo::getCudaCores( activeGPU ) },
       { "GPU clock rate (MHz)", (double) Devices::CudaDeviceInfo::getClockRate( activeGPU ) / 1e3 },
       { "GPU global memory (GB)", (double) Devices::CudaDeviceInfo::getGlobalMemory( activeGPU ) / 1e9 },
       { "GPU memory clock rate (MHz)", (double) Devices::CudaDeviceInfo::getMemoryClockRate( activeGPU ) / 1e3 },
       { "GPU memory ECC enabled", Devices::CudaDeviceInfo::getECCEnabled( activeGPU ) },
#endif
   };

   // TODO: implement resolveMatrixType
//   return ! Matrices::resolveMatrixType< MainConfig,
//                                         Devices::Host,
//                                         SpmvBenchmark >( benchmark, metadata, parameters );
   using MatrixType = Matrices::SlicedEllpack< double, Devices::Host, int >;
   const bool status = SpmvBenchmark< MatrixType >::run( benchmark, metadata, parameters );

   if( rank == 0 )
      if( ! benchmark.save( logFile ) ) {
         std::cerr << "Failed to write the benchmark results to file '" << parameters.getParameter< String >( "log-file" ) << "'." << std::endl;
         return EXIT_FAILURE;
      }

   return ! status;
}
