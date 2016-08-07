/***************************************************************************
                          tnlSolverConfig_impl.h  -  description
                             -------------------
    begin                : Jul 8, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/tnlConfig.h>
#include <TNL/Solvers/tnlBuildConfigTags.h>
#include <TNL/Solvers/tnlDummyProblem.h>
#include <TNL/Solvers/pde/tnlExplicitTimeStepper.h>
#include <TNL/Solvers/pde/tnlPDESolver.h>

namespace TNL {
namespace Solvers {

template< typename ConfigTag,
          typename ProblemConfig >
bool tnlSolverConfig< ConfigTag, ProblemConfig >::configSetup( Config::ConfigDescription& config )
{
   typedef tnlDummyProblem< double, Devices::Host, int > DummyProblem;

   config.addDelimiter( " === General parameters ==== " );
   /****
    * Setup real type
    */
   config.addEntry< String >( "real-type",
                                 "Precision of the floating point arithmetics.",
                                 "double" );
   if( tnlConfigTagReal< ConfigTag, float >::enabled )
      config.addEntryEnum( "float" );
   if( tnlConfigTagReal< ConfigTag, double >::enabled )
      config.addEntryEnum( "double" );
   if( tnlConfigTagReal< ConfigTag, long double >::enabled )
      config.addEntryEnum( "long-double" );

   /****
    * Setup device.
    */
   config.addEntry< String >( "device",
                                 "Device to use for the computations.",
                                 "host" );
   if( tnlConfigTagDevice< ConfigTag, Devices::Host >::enabled )
      config.addEntryEnum( "host" );
#ifdef HAVE_CUDA
   if( tnlConfigTagDevice< ConfigTag, Devices::Cuda >::enabled )
      config.addEntryEnum( "cuda" );
#endif

   /****
    * Setup index type.
    */
   config.addEntry< String >( "index-type",
                                 "Indexing type for arrays, vectors, matrices etc.",
                                 "int" );
   if( tnlConfigTagIndex< ConfigTag, short int >::enabled )
      config.addEntryEnum( "short-int" );

   if( tnlConfigTagIndex< ConfigTag, int >::enabled )
      config.addEntryEnum( "int" );

   if( tnlConfigTagIndex< ConfigTag, long int >::enabled )
      config.addEntryEnum( "long-int" );

   /****
    * Mesh file parameter
    */
   config.addDelimiter( " === Space discretisation parameters ==== " );
   config.addEntry< String >( "mesh", "A file which contains the numerical mesh. You may create it with tools like tnl-grid-setup or tnl-mesh-convert.", "mesh.tnl" );


   /****
    * Time discretisation
    */
   config.addDelimiter( " === Time discretisation parameters ==== " );
   typedef tnlExplicitTimeStepper< DummyProblem, tnlEulerSolver > ExplicitTimeStepper;
   tnlPDESolver< DummyProblem, ExplicitTimeStepper >::configSetup( config );
   ExplicitTimeStepper::configSetup( config );
   if( tnlConfigTagTimeDiscretisation< ConfigTag, tnlExplicitTimeDiscretisationTag >::enabled ||
       tnlConfigTagTimeDiscretisation< ConfigTag, tnlSemiImplicitTimeDiscretisationTag >::enabled ||
       tnlConfigTagTimeDiscretisation< ConfigTag, tnlImplicitTimeDiscretisationTag >::enabled )
   {
      config.addRequiredEntry< String >( "time-discretisation", "Discratisation in time.");
      if( tnlConfigTagTimeDiscretisation< ConfigTag, tnlExplicitTimeDiscretisationTag >::enabled )
         config.addEntryEnum( "explicit" );
      if( tnlConfigTagTimeDiscretisation< ConfigTag, tnlSemiImplicitTimeDiscretisationTag >::enabled )
         config.addEntryEnum( "semi-implicit" );
      if( tnlConfigTagTimeDiscretisation< ConfigTag, tnlImplicitTimeDiscretisationTag >::enabled )
         config.addEntryEnum( "implicit" );
   }
   config.addRequiredEntry< String >( "discrete-solver", "The solver of the discretised problem:" );
   if( tnlConfigTagTimeDiscretisation< ConfigTag, tnlExplicitTimeDiscretisationTag >::enabled )
   {
      if( tnlConfigTagExplicitSolver< ConfigTag, tnlExplicitEulerSolverTag >::enabled )
         config.addEntryEnum( "euler" );
      if( tnlConfigTagExplicitSolver< ConfigTag, tnlExplicitMersonSolverTag >::enabled )
         config.addEntryEnum( "merson" );
   }
   if( tnlConfigTagTimeDiscretisation< ConfigTag, tnlSemiImplicitTimeDiscretisationTag >::enabled )
   {
      if( tnlConfigTagSemiImplicitSolver< ConfigTag, tnlSemiImplicitCGSolverTag >::enabled )
         config.addEntryEnum( "cg" );
      if( tnlConfigTagSemiImplicitSolver< ConfigTag, tnlSemiImplicitBICGStabSolverTag >::enabled )
         config.addEntryEnum( "bicgstab" );
      if( tnlConfigTagSemiImplicitSolver< ConfigTag, tnlSemiImplicitGMRESSolverTag >::enabled )
         config.addEntryEnum( "gmres" );
      if( tnlConfigTagSemiImplicitSolver< ConfigTag, tnlSemiImplicitTFQMRSolverTag >::enabled )
         config.addEntryEnum( "tfqmr" );
      if( tnlConfigTagSemiImplicitSolver< ConfigTag, tnlSemiImplicitSORSolverTag >::enabled )
         config.addEntryEnum( "sor" );
#ifdef HAVE_UMFPACK
      if( tnlMeshConfigSemiImplicitSolver< MeshConfig, tnlSemiImplicitUmfpackSolverTag >::enabled )
         config.addEntryEnum( "umfpack" );
#endif
   }
   config.addEntry< String >( "preconditioner", "The preconditioner for the discrete solver:", "none" );
   config.addEntryEnum( "none" );
   config.addEntryEnum( "diagonal" );
   if( tnlConfigTagTimeDiscretisation< ConfigTag, tnlExplicitTimeDiscretisationTag >::enabled ||
       tnlConfigTagTimeDiscretisation< ConfigTag, tnlSemiImplicitTimeDiscretisationTag >::enabled )
   {
      config.addDelimiter( " === Iterative solvers parameters === " );
      tnlIterativeSolver< double, int >::configSetup( config );
   }
   if( tnlConfigTagTimeDiscretisation< ConfigTag, tnlExplicitTimeDiscretisationTag >::enabled )
   {
      config.addDelimiter( " === Explicit solvers parameters === " );
      tnlExplicitSolver< tnlDummyProblem< double, Devices::Host, int > >::configSetup( config );
      if( tnlConfigTagExplicitSolver< ConfigTag, tnlExplicitEulerSolverTag >::enabled )
         tnlEulerSolver< tnlDummyProblem< double, Devices::Host, int > >::configSetup( config );

      if( tnlConfigTagExplicitSolver< ConfigTag, tnlExplicitMersonSolverTag >::enabled )
         tnlMersonSolver< tnlDummyProblem< double, Devices::Host, int > >::configSetup( config );
   }
   if( tnlConfigTagTimeDiscretisation< ConfigTag, tnlSemiImplicitTimeDiscretisationTag >::enabled )
   {
      config.addDelimiter( " === Semi-implicit solvers parameters === " );
      typedef Matrices::CSRMatrix< double, Devices::Host, int > MatrixType;
      if( tnlConfigTagSemiImplicitSolver< ConfigTag, tnlSemiImplicitCGSolverTag >::enabled )
         Linear::Krylov::tnlCGSolver< MatrixType >::configSetup( config );
      if( tnlConfigTagSemiImplicitSolver< ConfigTag, tnlSemiImplicitBICGStabSolverTag >::enabled )
         Linear::Krylov::tnlBICGStabSolver< MatrixType >::configSetup( config );
      if( tnlConfigTagSemiImplicitSolver< ConfigTag, tnlSemiImplicitGMRESSolverTag >::enabled )
         Linear::Krylov::tnlGMRESSolver< MatrixType >::configSetup( config );
      if( tnlConfigTagSemiImplicitSolver< ConfigTag, tnlSemiImplicitTFQMRSolverTag >::enabled )
         Linear::Krylov::tnlTFQMRSolver< MatrixType >::configSetup( config );
      if( tnlConfigTagSemiImplicitSolver< ConfigTag, tnlSemiImplicitSORSolverTag >::enabled )
         Linear::tnlSORSolver< MatrixType >::configSetup( config );
   }

   config.addDelimiter( " === Logs and messages ===" );
   config.addEntry< int >( "verbose", "Set the verbose mode. The higher number the more messages are generated.", 1 );
   config.addEntry< String >( "log-file", "Log file for the computation.", "log.txt" );
   config.addEntry< int >( "log-width", "Number of columns of the log table.", 80 );
   return true;

}

} // namespace Solvers
} // namespace TNL

