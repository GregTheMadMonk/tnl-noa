/***************************************************************************
                          SolverConfig_impl.h  -  description
                             -------------------
    begin                : Jul 8, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/tnlConfig.h>
#include <TNL/Solvers/BuildConfigTags.h>
#include <TNL/Solvers/DummyProblem.h>
#include <TNL/Solvers/PDE/ExplicitTimeStepper.h>
#include <TNL/Solvers/PDE/PDESolver.h>

namespace TNL {
namespace Solvers {

template< typename ConfigTag,
          typename ProblemConfig >
bool SolverConfig< ConfigTag, ProblemConfig >::configSetup( Config::ConfigDescription& config )
{
   typedef DummyProblem< double, Devices::Host, int > DummyProblemType;

   config.addDelimiter( " === General parameters ==== " );
   /****
    * Setup real type
    */
   config.addEntry< String >( "real-type",
                                 "Precision of the floating point arithmetics.",
                                 "double" );
   if( ConfigTagReal< ConfigTag, float >::enabled )
      config.addEntryEnum( "float" );
   if( ConfigTagReal< ConfigTag, double >::enabled )
      config.addEntryEnum( "double" );
   if( ConfigTagReal< ConfigTag, long double >::enabled )
      config.addEntryEnum( "long-double" );

   /****
    * Setup device.
    */
   config.addEntry< String >( "device",
                                 "Device to use for the computations.",
                                 "host" );
   if( ConfigTagDevice< ConfigTag, Devices::Host >::enabled )
      config.addEntryEnum( "host" );
#ifdef HAVE_CUDA
   if( ConfigTagDevice< ConfigTag, Devices::Cuda >::enabled )
      config.addEntryEnum( "cuda" );
#endif

   /****
    * Setup index type.
    */
   config.addEntry< String >( "index-type",
                                 "Indexing type for arrays, vectors, matrices etc.",
                                 "int" );
   if( ConfigTagIndex< ConfigTag, short int >::enabled )
      config.addEntryEnum( "short-int" );

   if( ConfigTagIndex< ConfigTag, int >::enabled )
      config.addEntryEnum( "int" );

   if( ConfigTagIndex< ConfigTag, long int >::enabled )
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
   typedef PDE::ExplicitTimeStepper< DummyProblemType, ODE::Euler > ExplicitTimeStepper;
   PDE::PDESolver< DummyProblemType, ExplicitTimeStepper >::configSetup( config );
   ExplicitTimeStepper::configSetup( config );
   if( ConfigTagTimeDiscretisation< ConfigTag, tnlExplicitTimeDiscretisationTag >::enabled ||
       ConfigTagTimeDiscretisation< ConfigTag, tnlSemiImplicitTimeDiscretisationTag >::enabled ||
       ConfigTagTimeDiscretisation< ConfigTag, tnlImplicitTimeDiscretisationTag >::enabled )
   {
      config.addRequiredEntry< String >( "time-discretisation", "Discratisation in time.");
      if( ConfigTagTimeDiscretisation< ConfigTag, tnlExplicitTimeDiscretisationTag >::enabled )
         config.addEntryEnum( "explicit" );
      if( ConfigTagTimeDiscretisation< ConfigTag, tnlSemiImplicitTimeDiscretisationTag >::enabled )
         config.addEntryEnum( "semi-implicit" );
      if( ConfigTagTimeDiscretisation< ConfigTag, tnlImplicitTimeDiscretisationTag >::enabled )
         config.addEntryEnum( "implicit" );
   }
   config.addRequiredEntry< String >( "discrete-solver", "The solver of the discretised problem:" );
   if( ConfigTagTimeDiscretisation< ConfigTag, tnlExplicitTimeDiscretisationTag >::enabled )
   {
      if( ConfigTagExplicitSolver< ConfigTag, tnlExplicitEulerSolverTag >::enabled )
         config.addEntryEnum( "euler" );
      if( ConfigTagExplicitSolver< ConfigTag, tnlExplicitMersonSolverTag >::enabled )
         config.addEntryEnum( "merson" );
   }
   if( ConfigTagTimeDiscretisation< ConfigTag, tnlSemiImplicitTimeDiscretisationTag >::enabled )
   {
      if( ConfigTagSemiImplicitSolver< ConfigTag, tnlSemiImplicitCGSolverTag >::enabled )
         config.addEntryEnum( "cg" );
      if( ConfigTagSemiImplicitSolver< ConfigTag, tnlSemiImplicitBICGStabSolverTag >::enabled )
         config.addEntryEnum( "bicgstab" );
      if( ConfigTagSemiImplicitSolver< ConfigTag, tnlSemiImplicitGMRESSolverTag >::enabled )
         config.addEntryEnum( "gmres" );
      if( ConfigTagSemiImplicitSolver< ConfigTag, tnlSemiImplicitTFQMRSolverTag >::enabled )
         config.addEntryEnum( "tfqmr" );
      if( ConfigTagSemiImplicitSolver< ConfigTag, tnlSemiImplicitSORSolverTag >::enabled )
         config.addEntryEnum( "sor" );
#ifdef HAVE_UMFPACK
      if( MeshConfigSemiImplicitSolver< MeshConfig, tnlSemiImplicitUmfpackSolverTag >::enabled )
         config.addEntryEnum( "umfpack" );
#endif
   }
   config.addEntry< String >( "preconditioner", "The preconditioner for the discrete solver:", "none" );
   config.addEntryEnum( "none" );
   config.addEntryEnum( "diagonal" );
   if( ConfigTagTimeDiscretisation< ConfigTag, tnlExplicitTimeDiscretisationTag >::enabled ||
       ConfigTagTimeDiscretisation< ConfigTag, tnlSemiImplicitTimeDiscretisationTag >::enabled )
   {
      config.addDelimiter( " === Iterative solvers parameters === " );
      IterativeSolver< double, int >::configSetup( config );
   }
   if( ConfigTagTimeDiscretisation< ConfigTag, tnlExplicitTimeDiscretisationTag >::enabled )
   {
      config.addDelimiter( " === Explicit solvers parameters === " );
      ODE::ExplicitSolver< DummyProblem< double, Devices::Host, int > >::configSetup( config );
      if( ConfigTagExplicitSolver< ConfigTag, tnlExplicitEulerSolverTag >::enabled )
         ODE::Euler< DummyProblem< double, Devices::Host, int > >::configSetup( config );

      if( ConfigTagExplicitSolver< ConfigTag, tnlExplicitMersonSolverTag >::enabled )
         ODE::Merson< DummyProblem< double, Devices::Host, int > >::configSetup( config );
   }
   if( ConfigTagTimeDiscretisation< ConfigTag, tnlSemiImplicitTimeDiscretisationTag >::enabled )
   {
      config.addDelimiter( " === Semi-implicit solvers parameters === " );
      typedef Matrices::CSRMatrix< double, Devices::Host, int > MatrixType;
      if( ConfigTagSemiImplicitSolver< ConfigTag, tnlSemiImplicitCGSolverTag >::enabled )
         Linear::CG< MatrixType >::configSetup( config );
      if( ConfigTagSemiImplicitSolver< ConfigTag, tnlSemiImplicitBICGStabSolverTag >::enabled )
         Linear::BICGStab< MatrixType >::configSetup( config );
      if( ConfigTagSemiImplicitSolver< ConfigTag, tnlSemiImplicitGMRESSolverTag >::enabled )
         Linear::GMRES< MatrixType >::configSetup( config );
      if( ConfigTagSemiImplicitSolver< ConfigTag, tnlSemiImplicitTFQMRSolverTag >::enabled )
         Linear::TFQMR< MatrixType >::configSetup( config );
      if( ConfigTagSemiImplicitSolver< ConfigTag, tnlSemiImplicitSORSolverTag >::enabled )
         Linear::SOR< MatrixType >::configSetup( config );
   }

   config.addDelimiter( " === Logs and messages ===" );
   config.addEntry< int >( "verbose", "Set the verbose mode. The higher number the more messages are generated.", 1 );
   config.addEntry< String >( "log-file", "Log file for the computation.", "log.txt" );
   config.addEntry< int >( "log-width", "Number of columns of the log table.", 80 );
   return true;

}

} // namespace Solvers
} // namespace TNL

