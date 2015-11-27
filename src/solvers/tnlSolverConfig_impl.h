/***************************************************************************
                          tnlSolverConfig_impl.h  -  description
                             -------------------
    begin                : Jul 8, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TNLSOLVERCONFIG_IMPL_H_
#define TNLSOLVERCONFIG_IMPL_H_

#include <tnlConfig.h>
#include <solvers/tnlBuildConfigTags.h>
#include <solvers/tnlDummyProblem.h>
#include <solvers/pde/tnlExplicitTimeStepper.h>
#include <solvers/pde/tnlPDESolver.h>

template< typename MeshConfig,
          typename ProblemConfig >
bool tnlSolverConfig< MeshConfig, ProblemConfig >::configSetup( tnlConfigDescription& config )
{
   typedef tnlDummyProblem< double, tnlHost, int > DummyProblem;

   config.addDelimiter( " === General parameters ==== " );
   /****
    * Setup real type
    */
   config.addEntry< tnlString >( "real-type",
                                 "Precision of the floating point arithmetics.",
                                 "double" );
   if( tnlMeshConfigReal< MeshConfig, float >::enabled )
      config.addEntryEnum( "float" );
   if( tnlMeshConfigReal< MeshConfig, double >::enabled )
      config.addEntryEnum( "double" );
   if( tnlMeshConfigReal< MeshConfig, long double >::enabled )
      config.addEntryEnum( "long-double" );

   /****
    * Setup device.
    */
   config.addEntry< tnlString >( "device",
                                 "Device to use for the computations.",
                                 "host" );
   if( tnlMeshConfigDevice< MeshConfig, tnlHost >::enabled )
      config.addEntryEnum( "host" );
#ifdef HAVE_CUDA
   if( tnlMeshConfigDevice< MeshConfig, tnlCuda >::enabled )
      config.addEntryEnum( "cuda" );
#endif

   /****
    * Setup index type.
    */
   config.addEntry< tnlString >( "index-type",
                                 "Indexing type for arrays, vectors, matrices etc.",
                                 "int" );
   if( tnlMeshConfigIndex< MeshConfig, short int >::enabled )
      config.addEntryEnum( "short-int" );

   if( tnlMeshConfigIndex< MeshConfig, int >::enabled )
      config.addEntryEnum( "int" );

   if( tnlMeshConfigIndex< MeshConfig, long int >::enabled )
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
   if( tnlMeshConfigTimeDiscretisation< MeshConfig, tnlExplicitTimeDiscretisationTag >::enabled ||
       tnlMeshConfigTimeDiscretisation< MeshConfig, tnlSemiImplicitTimeDiscretisationTag >::enabled ||
       tnlMeshConfigTimeDiscretisation< MeshConfig, tnlImplicitTimeDiscretisationTag >::enabled )
   {
      config.addRequiredEntry< tnlString >( "time-discretisation", "Discratisation in time.");
      if( tnlMeshConfigTimeDiscretisation< MeshConfig, tnlExplicitTimeDiscretisationTag >::enabled )
         config.addEntryEnum( "explicit" );
      if( tnlMeshConfigTimeDiscretisation< MeshConfig, tnlSemiImplicitTimeDiscretisationTag >::enabled )
         config.addEntryEnum( "semi-implicit" );
      if( tnlMeshConfigTimeDiscretisation< MeshConfig, tnlImplicitTimeDiscretisationTag >::enabled )
         config.addEntryEnum( "implicit" );
   }
   config.addRequiredEntry< tnlString >( "discrete-solver", "The solver of the discretised problem:" );
   if( tnlMeshConfigTimeDiscretisation< MeshConfig, tnlExplicitTimeDiscretisationTag >::enabled )
   {      
      if( tnlMeshConfigExplicitSolver< MeshConfig, tnlExplicitEulerSolverTag >::enabled )
         config.addEntryEnum( "euler" );
      if( tnlMeshConfigExplicitSolver< MeshConfig, tnlExplicitMersonSolverTag >::enabled )
         config.addEntryEnum( "merson" );
   }
   if( tnlMeshConfigTimeDiscretisation< MeshConfig, tnlSemiImplicitTimeDiscretisationTag >::enabled )
   {
      if( tnlMeshConfigSemiImplicitSolver< MeshConfig, tnlSemiImplicitCGSolverTag >::enabled )
         config.addEntryEnum( "cg" );
      if( tnlMeshConfigSemiImplicitSolver< MeshConfig, tnlSemiImplicitBICGStabSolverTag >::enabled )
         config.addEntryEnum( "bicgstab" );
      if( tnlMeshConfigSemiImplicitSolver< MeshConfig, tnlSemiImplicitGMRESSolverTag >::enabled )
         config.addEntryEnum( "gmres" );
      if( tnlMeshConfigSemiImplicitSolver< MeshConfig, tnlSemiImplicitTFQMRSolverTag >::enabled )
         config.addEntryEnum( "tfqmr" );
      if( tnlMeshConfigSemiImplicitSolver< MeshConfig, tnlSemiImplicitSORSolverTag >::enabled )
         config.addEntryEnum( "sor" );
   }
   if( tnlMeshConfigTimeDiscretisation< MeshConfig, tnlExplicitTimeDiscretisationTag >::enabled ||
       tnlMeshConfigTimeDiscretisation< MeshConfig, tnlSemiImplicitTimeDiscretisationTag >::enabled )
   {
      config.addDelimiter( " === Iterative solvers parameters === " );
      tnlIterativeSolver< double, int >::configSetup( config );
   }
   if( tnlMeshConfigTimeDiscretisation< MeshConfig, tnlExplicitTimeDiscretisationTag >::enabled )
   {
      config.addDelimiter( " === Explicit solvers parameters === " );
      tnlExplicitSolver< tnlDummyProblem< double, tnlHost, int > >::configSetup( config );
      if( tnlMeshConfigExplicitSolver< MeshConfig, tnlExplicitEulerSolverTag >::enabled )
         tnlEulerSolver< tnlDummyProblem< double, tnlHost, int > >::configSetup( config );

      if( tnlMeshConfigExplicitSolver< MeshConfig, tnlExplicitMersonSolverTag >::enabled )
         tnlMersonSolver< tnlDummyProblem< double, tnlHost, int > >::configSetup( config );
   }
   if( tnlMeshConfigTimeDiscretisation< MeshConfig, tnlSemiImplicitTimeDiscretisationTag >::enabled )
   {
      config.addDelimiter( " === Semi-implicit solvers parameters === " );      
      typedef tnlCSRMatrix< double, tnlHost, int > MatrixType;
      if( tnlMeshConfigSemiImplicitSolver< MeshConfig, tnlSemiImplicitCGSolverTag >::enabled )
         tnlCGSolver< MatrixType >::configSetup( config );
      if( tnlMeshConfigSemiImplicitSolver< MeshConfig, tnlSemiImplicitBICGStabSolverTag >::enabled )
         tnlBICGStabSolver< MatrixType >::configSetup( config );
      if( tnlMeshConfigSemiImplicitSolver< MeshConfig, tnlSemiImplicitGMRESSolverTag >::enabled )
         tnlGMRESSolver< MatrixType >::configSetup( config );
      if( tnlMeshConfigSemiImplicitSolver< MeshConfig, tnlSemiImplicitTFQMRSolverTag >::enabled )
         tnlTFQMRSolver< MatrixType >::configSetup( config );
      if( tnlMeshConfigSemiImplicitSolver< MeshConfig, tnlSemiImplicitSORSolverTag >::enabled )
         tnlSORSolver< MatrixType >::configSetup( config );
   }

   config.addDelimiter( " === Logs and messages ===" );
   config.addEntry< int >( "verbose", "Set the verbose mode. The higher number the more messages are generated.", 1 );
   config.addEntry< tnlString >( "log-file", "Log file for the computation." );
   config.addEntry< int >( "log-width", "Number of columns of the log table.", 80 );
   return true;

}

#endif /* TNLSOLVERCONFIG_IMPL_H_ */
