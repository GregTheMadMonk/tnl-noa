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
#include <solvers/tnlConfigTags.h>
#include <solvers/tnlDummyProblem.h>

template< typename ConfigTag,
          typename ProblemConfig >
bool tnlSolverConfig< ConfigTag, ProblemConfig >::configSetup( tnlConfigDescription& config )
{
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
    * Mesh setup
    */
   config.addEntry< tnlString >( "mesh", "A file which contains the numerical mesh.", "mesh.tnl" );

   /****
    * Time discretisation
    */
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
   if( tnlConfigTagTimeDiscretisation< ConfigTag, tnlExplicitTimeDiscretisationTag >::enabled )
   {
      config.addDelimiter( " === Explicit solvers parameters === " );
      tnlEulerSolver< tnlDummyProblem< double, tnlHost, int > >::configSetup( config );

      tnlMersonSolver< tnlDummyProblem< double, tnlHost, int > >::configSetup( config );
   }

   config.addEntry< int >( "verbose", "Set the verbose mode. The higher number the more messages are generated.", 1 );
   return true;

}

#endif /* TNLSOLVERCONFIG_IMPL_H_ */
