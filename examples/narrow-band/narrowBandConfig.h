/***************************************************************************
                          narrowBandConfig.h  -  description
                             -------------------
    begin                : Oct 15, 2015
    copyright            : (C) 2015 by Tomas Sobotik
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

#ifndef NARROWBANDCONFIG_H_
#define NARROWBANDCONFIG_H_

#include <config/tnlConfigDescription.h>

template< typename ConfigTag >
class narrowBandConfig
{
   public:
      static void configSetup( tnlConfigDescription& config )
      {
         config.addDelimiter( "Parallel Eikonal solver settings:" );
         config.addEntry        < tnlString > ( "problem-name", "This defines particular problem.", "fast-sweeping" );
         config.addRequiredEntry        < tnlString > ( "initial-condition", "Initial condition for solver");
         config.addRequiredEntry        < int > ( "dim", "Dimension of problem.");
         config.addRequiredEntry        < double > ( "tau", "Time step.");
         config.addRequiredEntry        < double > ( "final-time", "Final time.");
         config.addEntry       < tnlString > ( "mesh", "Name of mesh.", "mesh.tnl" );
         config.addEntry       < tnlString > ( "exact-input", "Are the function values near the curve equal to the SDF? (yes/no)", "no" );
      }
};

#endif /* NARROWBANDCONFIG_H_ */
