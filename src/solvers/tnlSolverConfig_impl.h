/***************************************************************************
                          tnlSolverConfig_impl.h  -  description
                             -------------------
    begin                : Jul 8, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLSOLVERCONFIG_IMPL_H_
#define TNLSOLVERCONFIG_IMPL_H_

#include <tnlConfig.h>
#include <solvers/tnlBuildConfigTags.h>
#include <solvers/tnlDummyProblem.h>
#include <solvers/pde/tnlExplicitTimeStepper.h>
#include <solvers/pde/tnlPDESolver.h>

template< typename ConfigTag,
          typename ProblemConfig >
bool tnlSolverConfig< ConfigTag, ProblemConfig >::configSetup( tnlConfigDescription& config )
{
   typedef tnlDummyProblem< double, tnlHost, int > DummyProblem;

   config.addDelimiter( " === General parameters ==== " );
   /****
    * Setup real type
    */
   config.addEntry< tnlString >( "real-type",
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
   config.addEntry< tnlString >( "device",
                                 "Device to use for the computations.",
                                 "host" );
   if( tnlConfigTagDevice< ConfigTag, tnlHost >::enabled )
      config.addEntryEnum( "host" );
#ifdef HAVE_CUDA
   if( tnlConfigTagDevice< ConfigTag, tnlCuda >::enabled )
      config.addEntryEnum( "cuda" );
#endif

   /****
    * Setup index type.
    */
   config.addEntry< tnlString >( "index-type",
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
   config.addEntry< tnlString >( "mesh", "A file which contains the numerical mesh. You may create it with tools like tnl-grid-setup or tnl-mesh-convert.", "mesh.tnl" );


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
      config.addRequiredEntry< tnlString >( "time-discretisation", "Discratisation in time.");
      if( tnlConfigTagTimeDiscretisation< ConfigTag, tnlExplicitTimeDiscretisationTag >::enabled )
         config.addEntryEnum( "explicit" );
      if( tnlConfigTagTimeDiscretisation< ConfigTag, tnlSemiImplicitTimeDiscretisationTag >::enabled )
         config.addEntryEnum( "semi-implicit" );
      if( tnlConfigTagTimeDiscretisation< ConfigTag, tnlImplicitTimeDiscretisationTag >::enabled )
         config.addEntryEnum( "implicit" );
   }
   config.addRequiredEntry< tnlString >( "discrete-solver", "The solver of the discretised problem:" );
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
   config.addEntry< tnlString >( "preconditioner", "The preconditioner for the discrete solver:", "none" );
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
      tnlExplicitSolver< tnlDummyProblem< double, tnlHost, int > >::configSetup( config );
      if( tnlConfigTagExplicitSolver< ConfigTag, tnlExplicitEulerSolverTag >::enabled )
         tnlEulerSolver< tnlDummyProblem< double, tnlHost, int > >::configSetup( config );

      if( tnlConfigTagExplicitSolver< ConfigTag, tnlExplicitMersonSolverTag >::enabled )
         tnlMersonSolver< tnlDummyProblem< double, tnlHost, int > >::configSetup( config );
   }
   if( tnlConfigTagTimeDiscretisation< ConfigTag, tnlSemiImplicitTimeDiscretisationTag >::enabled )
   {
      config.addDelimiter( " === Semi-implicit solvers parameters === " );
      typedef tnlCSRMatrix< double, tnlHost, int > MatrixType;
      if( tnlConfigTagSemiImplicitSolver< ConfigTag, tnlSemiImplicitCGSolverTag >::enabled )
         tnlCGSolver< MatrixType >::configSetup( config );
      if( tnlConfigTagSemiImplicitSolver< ConfigTag, tnlSemiImplicitBICGStabSolverTag >::enabled )
         tnlBICGStabSolver< MatrixType >::configSetup( config );
      if( tnlConfigTagSemiImplicitSolver< ConfigTag, tnlSemiImplicitGMRESSolverTag >::enabled )
         tnlGMRESSolver< MatrixType >::configSetup( config );
      if( tnlConfigTagSemiImplicitSolver< ConfigTag, tnlSemiImplicitTFQMRSolverTag >::enabled )
         tnlTFQMRSolver< MatrixType >::configSetup( config );
      if( tnlConfigTagSemiImplicitSolver< ConfigTag, tnlSemiImplicitSORSolverTag >::enabled )
         tnlSORSolver< MatrixType >::configSetup( config );
   }

   config.addDelimiter( " === Logs and messages ===" );
   config.addEntry< int >( "verbose", "Set the verbose mode. The higher number the more messages are generated.", 1 );
   config.addEntry< tnlString >( "log-file", "Log file for the computation.", "log.txt" );
   config.addEntry< int >( "log-width", "Number of columns of the log table.", 80 );
   return true;

}

#endif /* TNLSOLVERCONFIG_IMPL_H_ */
