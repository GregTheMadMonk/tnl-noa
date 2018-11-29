/***************************************************************************
                          parallelEikonalConfig.h  -  description
                             -------------------
    begin                : Oct 5, 2014
    copyright            : (C) 2014 by Tomas Sobotik
    email                :
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef HAMILTONJACOBIPARALLELEIKONALPROBLEMCONFIG_H_
#define HAMILTONJACOBIPARALLELEIKONALPROBLEMCONFIG_H_

#include <config/tnlConfigDescription.h>

template< typename ConfigTag >
class parallelEikonalConfig
{
   public:
      static void configSetup( tnlConfigDescription& config )
      {
         config.addDelimiter( "Parallel Eikonal solver settings:" );
         config.addEntry        < String > ( "problem-name", "This defines particular problem.", "hamilton-jacobi-parallel" );
         config.addEntry       < String > ( "scheme", "This defines scheme used for discretization.", "godunov" );
         config.addEntryEnum( "godunov" );
         config.addEntryEnum( "upwind" );
         config.addRequiredEntry        < String > ( "initial-condition", "Initial condition for solver");
         config.addEntry       < String > ( "mesh", "Name of mesh.", "mesh.tnl" );
         config.addEntry        < double > ( "epsilon", "This defines epsilon for smoothening of sign().", 0.0 );
         config.addEntry        < double > ( "delta", " Allowed difference on subgrid boundaries", 0.0 );
         config.addRequiredEntry        < double > ( "stop-time", " Final time for solver");
         config.addRequiredEntry        < double > ( "initial-tau", " initial tau for solver" );
         config.addEntry        < double > ( "cfl-condition", " CFL condition", 0.0 );
         config.addEntry        < int > ( "subgrid-size", "Subgrid size.", 16 );
         config.addRequiredEntry        < int > ( "dim", "Dimension of problem.");
      }
};

#endif /* HAMILTONJACOBIPARALLELEIKONALPROBLEMCONFIG_H_ */
