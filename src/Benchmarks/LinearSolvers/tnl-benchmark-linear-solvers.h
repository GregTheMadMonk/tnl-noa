/***************************************************************************
                          tnl-benchmark-linear-solvers.h  -  description
                             -------------------
    begin                : Sep 18, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovsky

#pragma once

#include <set>
#include <sstream>
#include <string>

#ifndef NDEBUG
#include <TNL/Debugging/FPE.h>
#endif

#include <TNL/Config/ConfigDescription.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Communicators/MpiCommunicator.h>
#include <TNL/Communicators/NoDistrCommunicator.h>
#include <TNL/Communicators/ScopedInitializer.h>
#include <TNL/Containers/Partitioner.h>
#include <TNL/Containers/DistributedVector.h>
#include <TNL/Matrices/DistributedMatrix.h>
#include <TNL/Solvers/Linear/Preconditioners/Diagonal.h>
#include <TNL/Solvers/Linear/Preconditioners/ILU0.h>
#include <TNL/Solvers/Linear/Preconditioners/ILUT.h>
#include <TNL/Solvers/Linear/GMRES.h>
#include <TNL/Solvers/Linear/TFQMR.h>
#include <TNL/Solvers/Linear/BICGStab.h>
#include <TNL/Solvers/Linear/BICGStabL.h>
#include <TNL/Solvers/Linear/UmfpackWrapper.h>

#include "../Benchmarks.h"
#include "../DistSpMV/ordering.h"
#include "benchmarks.h"

// FIXME: nvcc 8.0 fails when cusolverSp.h is included (works fine with clang):
// /opt/cuda/include/cuda_fp16.h(3068): error: more than one instance of overloaded function "isinf" matches the argument list:
//             function "isinf(float)"
//             function "std::isinf(float)"
//             argument types are: (float)
#if defined(HAVE_CUDA) && !defined(__NVCC__)
   #include "CuSolverWrapper.h"
   #define HAVE_CUSOLVER
#endif

#include <TNL/Matrices/SlicedEllpack.h>

using namespace TNL;
using namespace TNL::Benchmarks;
using namespace TNL::Pointers;

#ifdef HAVE_MPI
using CommunicatorType = Communicators::MpiCommunicator;
#else
using CommunicatorType = Communicators::NoDistrCommunicator;
#endif


static const std::set< std::string > valid_solvers = {
   "gmres",
   "cwygmres",
   "tfqmr",
   "bicgstab",
   "bicgstab-ell",
};

static const std::set< std::string > valid_preconditioners = {
   "jacobi",
   "ilu0",
   "ilut",
};

std::set< std::string >
parse_comma_list( const Config::ParameterContainer& parameters,
                  const char* parameter,
                  const std::set< std::string >& options )
{
   const String solvers = parameters.getParameter< String >( parameter );

   if( solvers == "all" )
      return options;

   std::stringstream ss( solvers.getString() );
   std::string s;
   std::set< std::string > set;

   while( std::getline( ss, s, ',' ) ) {
      if( ! options.count( s ) )
         throw std::logic_error( std::string("Invalid value in the comma-separated list for the parameter '")
                                 + parameter + "': '" + s + "'. The list contains: '" + solvers.getString() + "'." );

      set.insert( s );

      if( ss.peek() == ',' )
         ss.ignore();
   }

   return set;
}

template< typename Matrix, typename Vector >
void
benchmarkIterativeSolvers( Benchmark& benchmark,
// FIXME: ParameterContainer should be copyable, but that leads to double-free
//                           Config::ParameterContainer parameters,
                           Config::ParameterContainer& parameters,
                           const SharedPointer< Matrix >& matrixPointer,
                           const Vector& x0,
                           const Vector& b )
{
#ifdef HAVE_CUDA
   using CudaMatrix = typename Matrix::CudaType;
   using CudaVector = typename Vector::CudaType;

   CudaVector cuda_x0, cuda_b;
   cuda_x0 = x0;
   cuda_b = b;

   SharedPointer< CudaMatrix > cudaMatrixPointer;
   *cudaMatrixPointer = *matrixPointer;

   // synchronize shared pointers
   Devices::Cuda::synchronizeDevice();
#endif

   using namespace Solvers::Linear;
   using namespace Solvers::Linear::Preconditioners;

   const int ell_max = 2;
   const std::set< std::string > solvers = parse_comma_list( parameters, "solvers", valid_solvers );
   const std::set< std::string > preconditioners = parse_comma_list( parameters, "preconditioners", valid_preconditioners );
   const bool with_preconditioner_update = parameters.getParameter< bool >( "with-preconditioner-update" );

   if( preconditioners.count( "jacobi" ) ) {
      if( with_preconditioner_update ) {
         benchmark.setOperation("preconditioner update (Jacobi)");
         benchmarkPreconditionerUpdate< Diagonal >( benchmark, parameters, matrixPointer );
         #ifdef HAVE_CUDA
         benchmarkPreconditionerUpdate< Diagonal >( benchmark, parameters, cudaMatrixPointer );
         #endif
      }

      if( solvers.count( "gmres" ) ) {
         benchmark.setOperation("GMRES (Jacobi)");
         parameters.template setParameter< String >( "gmres-variant", "MGSR" );
         benchmarkSolver< GMRES, Diagonal >( benchmark, parameters, matrixPointer, x0, b );
         #ifdef HAVE_CUDA
         benchmarkSolver< GMRES, Diagonal >( benchmark, parameters, cudaMatrixPointer, cuda_x0, cuda_b );
         #endif
      }

      if( solvers.count( "cwygmres" ) ) {
         benchmark.setOperation("CWYGMRES (Jacobi)");
         parameters.template setParameter< String >( "gmres-variant", "CWY" );
         benchmarkSolver< GMRES, Diagonal >( benchmark, parameters, matrixPointer, x0, b );
         #ifdef HAVE_CUDA
         benchmarkSolver< GMRES, Diagonal >( benchmark, parameters, cudaMatrixPointer, cuda_x0, cuda_b );
         #endif
      }

      if( solvers.count( "tfqmr" ) ) {
         benchmark.setOperation("TFQMR (Jacobi)");
         benchmarkSolver< TFQMR, Diagonal >( benchmark, parameters, matrixPointer, x0, b );
         #ifdef HAVE_CUDA
         benchmarkSolver< TFQMR, Diagonal >( benchmark, parameters, cudaMatrixPointer, cuda_x0, cuda_b );
         #endif
      }

      if( solvers.count( "bicgstab" ) ) {
         benchmark.setOperation("BiCGstab (Jacobi)");
         benchmarkSolver< BICGStab, Diagonal >( benchmark, parameters, matrixPointer, x0, b );
         #ifdef HAVE_CUDA
         benchmarkSolver< BICGStab, Diagonal >( benchmark, parameters, cudaMatrixPointer, cuda_x0, cuda_b );
         #endif
      }

      if( solvers.count( "bicgstab-ell" ) ) {
         benchmark.setOperation("BiCGstab (Jacobi)");
         for( int ell = 1; ell <= ell_max; ell++ ) {
            parameters.template setParameter< int >( "bicgstab-ell", ell );
            benchmark.setOperation("BiCGstab(" + convertToString(ell) + ") (Jacobi)");
            benchmarkSolver< BICGStabL, Diagonal >( benchmark, parameters, matrixPointer, x0, b );
            #ifdef HAVE_CUDA
            benchmarkSolver< BICGStabL, Diagonal >( benchmark, parameters, cudaMatrixPointer, cuda_x0, cuda_b );
            #endif
         }
      }
   }


   if( preconditioners.count( "ilu0" ) ) {
      if( with_preconditioner_update ) {
         benchmark.setOperation("preconditioner update (ILU0)");
         benchmarkPreconditionerUpdate< ILU0 >( benchmark, parameters, matrixPointer );
         #ifdef HAVE_CUDA
         benchmarkPreconditionerUpdate< ILU0 >( benchmark, parameters, cudaMatrixPointer );
         #endif
      }

      if( solvers.count( "gmres" ) ) {
         benchmark.setOperation("GMRES (ILU0)");
         parameters.template setParameter< String >( "gmres-variant", "MGSR" );
         benchmarkSolver< GMRES, ILU0 >( benchmark, parameters, matrixPointer, x0, b );
         #ifdef HAVE_CUDA
         benchmarkSolver< GMRES, ILU0 >( benchmark, parameters, cudaMatrixPointer, cuda_x0, cuda_b );
         #endif
      }

      if( solvers.count( "cwygmres" ) ) {
         benchmark.setOperation("CWYGMRES (ILU0)");
         parameters.template setParameter< String >( "gmres-variant", "CWY" );
         benchmarkSolver< GMRES, ILU0 >( benchmark, parameters, matrixPointer, x0, b );
         #ifdef HAVE_CUDA
         benchmarkSolver< GMRES, ILU0 >( benchmark, parameters, cudaMatrixPointer, cuda_x0, cuda_b );
         #endif
      }

      if( solvers.count( "tfqmr" ) ) {
         benchmark.setOperation("TFQMR (ILU0)");
         benchmarkSolver< TFQMR, ILU0 >( benchmark, parameters, matrixPointer, x0, b );
         #ifdef HAVE_CUDA
         benchmarkSolver< TFQMR, ILU0 >( benchmark, parameters, cudaMatrixPointer, cuda_x0, cuda_b );
         #endif
      }

      if( solvers.count( "bicgstab" ) ) {
         benchmark.setOperation("BiCGstab (ILU0)");
         benchmarkSolver< BICGStab, ILU0 >( benchmark, parameters, matrixPointer, x0, b );
         #ifdef HAVE_CUDA
         benchmarkSolver< BICGStab, ILU0 >( benchmark, parameters, cudaMatrixPointer, cuda_x0, cuda_b );
         #endif
      }

      if( solvers.count( "bicgstab-ell" ) ) {
         for( int ell = 1; ell <= ell_max; ell++ ) {
            parameters.template setParameter< int >( "bicgstab-ell", ell );
            benchmark.setOperation("BiCGstab(" + convertToString(ell) + ") (ILU0)");
            benchmarkSolver< BICGStabL, ILU0 >( benchmark, parameters, matrixPointer, x0, b );
            #ifdef HAVE_CUDA
            benchmarkSolver< BICGStabL, ILU0 >( benchmark, parameters, cudaMatrixPointer, cuda_x0, cuda_b );
            #endif
         }
      }
   }


   if( preconditioners.count( "ilut" ) ) {
      if( with_preconditioner_update ) {
         benchmark.setOperation("preconditioner update (ILUT)");
         benchmarkPreconditionerUpdate< ILUT >( benchmark, parameters, matrixPointer );
         #ifdef HAVE_CUDA
         benchmarkPreconditionerUpdate< ILUT >( benchmark, parameters, cudaMatrixPointer );
         #endif
      }

      if( solvers.count( "gmres" ) ) {
         benchmark.setOperation("GMRES (ILUT)");
         parameters.template setParameter< String >( "gmres-variant", "MGSR" );
         benchmarkSolver< GMRES, ILUT >( benchmark, parameters, matrixPointer, x0, b );
         #ifdef HAVE_CUDA
         benchmarkSolver< GMRES, ILUT >( benchmark, parameters, cudaMatrixPointer, cuda_x0, cuda_b );
         #endif
      }

      if( solvers.count( "cwygmres" ) ) {
         benchmark.setOperation("CWYGMRES (ILUT)");
         parameters.template setParameter< String >( "gmres-variant", "CWY" );
         benchmarkSolver< GMRES, ILUT >( benchmark, parameters, matrixPointer, x0, b );
         #ifdef HAVE_CUDA
         benchmarkSolver< GMRES, ILUT >( benchmark, parameters, cudaMatrixPointer, cuda_x0, cuda_b );
         #endif
      }

      if( solvers.count( "tfqmr" ) ) {
         benchmark.setOperation("TFQMR (ILUT)");
         benchmarkSolver< TFQMR, ILUT >( benchmark, parameters, matrixPointer, x0, b );
         #ifdef HAVE_CUDA
         benchmarkSolver< TFQMR, ILUT >( benchmark, parameters, cudaMatrixPointer, cuda_x0, cuda_b );
         #endif
      }

      if( solvers.count( "bicgstab" ) ) {
         benchmark.setOperation("BiCGstab (ILUT)");
         benchmarkSolver< BICGStab, ILUT >( benchmark, parameters, matrixPointer, x0, b );
         #ifdef HAVE_CUDA
         benchmarkSolver< BICGStab, ILUT >( benchmark, parameters, cudaMatrixPointer, cuda_x0, cuda_b );
         #endif
      }

      if( solvers.count( "bicgstab-ell" ) ) {
         for( int ell = 1; ell <= ell_max; ell++ ) {
            parameters.template setParameter< int >( "bicgstab-ell", ell );
            benchmark.setOperation("BiCGstab(" + convertToString(ell) + ") (ILUT)");
            benchmarkSolver< BICGStabL, ILUT >( benchmark, parameters, matrixPointer, x0, b );
            #ifdef HAVE_CUDA
            benchmarkSolver< BICGStabL, ILUT >( benchmark, parameters, cudaMatrixPointer, cuda_x0, cuda_b );
            #endif
         }
      }
   }
}

template< typename MatrixType >
struct LinearSolversBenchmark
{
   using RealType = typename MatrixType::RealType;
   using DeviceType = typename MatrixType::DeviceType;
   using IndexType = typename MatrixType::IndexType;
   using VectorType = Containers::Vector< RealType, DeviceType, IndexType >;

   using Partitioner = Containers::Partitioner< IndexType, CommunicatorType >;
   using DistributedMatrix = Matrices::DistributedMatrix< MatrixType, CommunicatorType >;
   using DistributedVector = Containers::DistributedVector< RealType, DeviceType, IndexType, CommunicatorType >;
   using DistributedRowLengths = typename DistributedMatrix::CompressedRowLengthsVector;

   static bool
   run( Benchmark& benchmark,
        Benchmark::MetadataMap metadata,
// FIXME: ParameterContainer should be copyable, but that leads to double-free
//        const Config::ParameterContainer& parameters )
        Config::ParameterContainer& parameters )
   {
      SharedPointer< MatrixType > matrixPointer;
      VectorType x0, b;
      matrixPointer->load( parameters.getParameter< String >( "input-matrix" ) );
      x0.load( parameters.getParameter< String >( "input-dof" ) );
      b.load( parameters.getParameter< String >( "input-rhs" ) );

      typename MatrixType::CompressedRowLengthsVector rowLengths;
      matrixPointer->getCompressedRowLengths( rowLengths );
      const IndexType maxRowLength = rowLengths.max();

      const String name = String( (CommunicatorType::isDistributed()) ? "Distributed linear solvers" : "Linear solvers" )
                          + " (" + parameters.getParameter< String >( "name" ) + "): ";
      benchmark.newBenchmark( name, metadata );
      benchmark.setMetadataColumns( Benchmark::MetadataColumns({
         // TODO: strip the device
//         { "matrix type", matrixPointer->getType() },
         { "rows", convertToString( matrixPointer->getRows() ) },
         { "columns", convertToString( matrixPointer->getColumns() ) },
         // FIXME: getMaxRowLengths() returns 0 for matrices loaded from file
//         { "max elements per row", matrixPointer->getMaxRowLength() },
         { "max elements per row", convertToString( maxRowLength ) },
      } ));

      const bool reorder = parameters.getParameter< bool >( "reorder-dofs" );
      if( reorder ) {
         using PermutationVector = Containers::Vector< IndexType, DeviceType, IndexType >;
         PermutationVector perm, iperm;
         getTrivialOrdering( *matrixPointer, perm, iperm );
         SharedPointer< MatrixType > matrix_perm;
         VectorType x0_perm, b_perm;
         x0_perm.setLike( x0 );
         b_perm.setLike( b );
         Matrices::reorderSparseMatrix( *matrixPointer, *matrix_perm, perm, iperm );
         Matrices::reorderArray( x0, x0_perm, perm );
         Matrices::reorderArray( b, b_perm, perm );
         if( CommunicatorType::isDistributed() )
            runDistributed( benchmark, metadata, parameters, matrix_perm, x0_perm, b_perm );
         else
            runNonDistributed( benchmark, metadata, parameters, matrix_perm, x0_perm, b_perm );
      }
      else {
         if( CommunicatorType::isDistributed() )
            runDistributed( benchmark, metadata, parameters, matrixPointer, x0, b );
         else
            runNonDistributed( benchmark, metadata, parameters, matrixPointer, x0, b );
      }

      return true;
   }

   static void
   runDistributed( Benchmark& benchmark,
                   Benchmark::MetadataMap metadata,
// FIXME: ParameterContainer should be copyable, but that leads to double-free
//                   const Config::ParameterContainer& parameters,
                   Config::ParameterContainer& parameters,
                   const SharedPointer< MatrixType >& matrixPointer,
                   const VectorType& x0,
                   const VectorType& b )
   {
      // set up the distributed matrix
      const auto group = CommunicatorType::AllGroup;
      const auto localRange = Partitioner::splitRange( matrixPointer->getRows(), group );
      SharedPointer< DistributedMatrix > distMatrixPointer( localRange, matrixPointer->getRows(), matrixPointer->getColumns(), group );
      DistributedVector dist_x0( localRange, matrixPointer->getRows(), group );
      DistributedVector dist_b( localRange, matrixPointer->getRows(), group );

      // copy the row lengths from the global matrix to the distributed matrix
      DistributedRowLengths distributedRowLengths( localRange, matrixPointer->getRows(), group );
      for( IndexType i = 0; i < distMatrixPointer->getLocalMatrix().getRows(); i++ ) {
         const auto gi = distMatrixPointer->getLocalRowRange().getGlobalIndex( i );
         distributedRowLengths[ gi ] = matrixPointer->getRowLength( gi );
      }
      distMatrixPointer->setCompressedRowLengths( distributedRowLengths );

      // copy data from the global matrix/vector into the distributed matrix/vector
      for( IndexType i = 0; i < distMatrixPointer->getLocalMatrix().getRows(); i++ ) {
         const auto gi = distMatrixPointer->getLocalRowRange().getGlobalIndex( i );
         dist_x0[ gi ] = x0[ gi ];
         dist_b[ gi ] = b[ gi ];

         const IndexType rowLength = matrixPointer->getRowLength( i );
         IndexType columns[ rowLength ];
         RealType values[ rowLength ];
         matrixPointer->getRowFast( gi, columns, values );
         distMatrixPointer->setRowFast( gi, columns, values, rowLength );
      }

      std::cout << "Iterative solvers:" << std::endl;
      benchmarkIterativeSolvers( benchmark, parameters, distMatrixPointer, dist_x0, dist_b );
   }

   static void
   runNonDistributed( Benchmark& benchmark,
                      Benchmark::MetadataMap metadata,
// FIXME: ParameterContainer should be copyable, but that leads to double-free
//                      const Config::ParameterContainer& parameters,
                      Config::ParameterContainer& parameters,
                      const SharedPointer< MatrixType >& matrixPointer,
                      const VectorType& x0,
                      const VectorType& b )
   {
      // direct solvers
      if( parameters.getParameter< bool >( "with-direct" ) ) {
         using CSR = Matrices::CSR< RealType, DeviceType, IndexType >;
         SharedPointer< CSR > matrixCopy;
         Matrices::copySparseMatrix( *matrixCopy, *matrixPointer );

#ifdef HAVE_UMFPACK
         std::cout << "UMFPACK wrapper:" << std::endl;
         using UmfpackSolver = Solvers::Linear::UmfpackWrapper< CSR >;
         using Preconditioner = Solvers::Linear::Preconditioners::Preconditioner< CSR >;
         benchmarkSolver< UmfpackSolver, Preconditioner >( parameters, matrixCopy, x0, b );
#endif

#ifdef HAVE_ARMADILLO
         std::cout << "Armadillo wrapper (which wraps SuperLU):" << std::endl;
         benchmarkArmadillo( parameters, matrixCopy, x0, b );
#endif
      }

      std::cout << "Iterative solvers:" << std::endl;
      benchmarkIterativeSolvers( benchmark, parameters, matrixPointer, x0, b );

#ifdef HAVE_CUSOLVER
      std::cout << "CuSOLVER:" << std::endl;
      {
         using CSR = Matrices::CSR< RealType, DeviceType, IndexType >;
         SharedPointer< CSR > matrixCopy;
         Matrices::copySparseMatrix( *matrixCopy, *matrixPointer );

         SharedPointer< typename CSR::CudaType > cuda_matrixCopy;
         *cuda_matrixCopy = *matrixCopy;
         typename VectorType::CudaType cuda_x0, cuda_b;
         cuda_x0.setLike( x0 );
         cuda_b.setLike( b );
         cuda_x0 = x0;
         cuda_b = b;

         using namespace Solvers::Linear;
         using namespace Solvers::Linear::Preconditioners;
         benchmarkSolver< CuSolverWrapper, Preconditioner >( benchmark, parameters, cuda_matrixCopy, cuda_x0, cuda_b );
      }
#endif
   }
};

void
configSetup( Config::ConfigDescription& config )
{
   config.addDelimiter( "Benchmark settings:" );
   config.addEntry< String >( "log-file", "Log file name.", "tnl-benchmark-linear-solvers.log");
   config.addEntry< String >( "output-mode", "Mode for opening the log file.", "overwrite" );
   config.addEntryEnum( "append" );
   config.addEntryEnum( "overwrite" );
   config.addEntry< int >( "loops", "Number of repetitions of the benchmark.", 10 );
   config.addRequiredEntry< String >( "input-matrix", "File name of the input matrix (in binary TNL format)." );
   config.addRequiredEntry< String >( "input-dof", "File name of the input DOF vector (in binary TNL format)." );
   config.addRequiredEntry< String >( "input-rhs", "File name of the input right-hand-side vector (in binary TNL format)." );
   config.addEntry< String >( "name", "Name of the matrix in the benchmark.", "" );
   config.addEntry< int >( "verbose", "Verbose mode.", 1 );
   config.addEntry< bool >( "reorder-dofs", "Reorder matrix entries corresponding to the same DOF together.", false );
   config.addEntry< bool >( "with-direct", "Includes the 3rd party direct solvers in the benchmark.", false );
   config.addEntry< String >( "solvers", "Comma-separated list of solvers to run benchmarks for. Options: gmres, cwygmres, tfqmr, bicgstab, bicgstab-ell.", "all" );
   config.addEntry< String >( "preconditioners", "Comma-separated list of preconditioners to run benchmarks for. Options: jacobi, ilu0, ilut.", "all" );
   config.addEntry< bool >( "with-preconditioner-update", "Run benchmark for the preconditioner update.", true );
   config.addEntry< String >( "devices", "Run benchmarks on these devices.", "all" );
   config.addEntryEnum( "all" );
   config.addEntryEnum( "host" );
   #ifdef HAVE_CUDA
   config.addEntryEnum( "cuda" );
   #endif

   config.addDelimiter( "Device settings:" );
   Devices::Host::configSetup( config );
   Devices::Cuda::configSetup( config );
   CommunicatorType::configSetup( config );

   config.addDelimiter( "Linear solver settings:" );
   Solvers::IterativeSolver< double, int >::configSetup( config );
   using Matrix = Matrices::SlicedEllpack< double, Devices::Host, int >;
   using GMRES = Solvers::Linear::GMRES< Matrix >;
   GMRES::configSetup( config );
   using BiCGstabL = Solvers::Linear::BICGStabL< Matrix >;
   BiCGstabL::configSetup( config );
   using ILUT = Solvers::Linear::Preconditioners::ILUT< Matrix >;
   ILUT::configSetup( config );
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
//                                         LinearSolversBenchmark >( benchmark, metadata, parameters );
   using MatrixType = Matrices::SlicedEllpack< double, Devices::Host, int >;
   const bool status = LinearSolversBenchmark< MatrixType >::run( benchmark, metadata, parameters );

   if( rank == 0 )
      if( ! benchmark.save( logFile ) ) {
         std::cerr << "Failed to write the benchmark results to file '" << parameters.getParameter< String >( "log-file" ) << "'." << std::endl;
         return EXIT_FAILURE;
      }

   return ! status;
}
