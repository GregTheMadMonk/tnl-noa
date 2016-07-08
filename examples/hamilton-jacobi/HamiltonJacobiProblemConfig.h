/***************************************************************************
                          hamiltonJacobiProblemConfig.h  -  description
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

#pragma once 

#include <config/tnlConfigDescription.h>

template< typename ConfigTag >
class HamiltonJacobiProblemConfig
{
   public:
      static void configSetup( tnlConfigDescription& config )
      {
         config.addDelimiter( "Hamilton-Jacobi solver settings:" );
         config.addEntry        < tnlString > ( "problem-name", "This defines particular problem.", "hamilton-jacobi" );
         config.addEntry       < tnlString > ( "scheme", "This defines scheme used for discretization.", "upwind" );
         config.addEntryEnum( "godunov" );
         config.addEntryEnum( "upwind" );
         config.addEntryEnum( "godunov2" );
         config.addEntryEnum( "upwind2" );
         config.addEntry        < double > ( "epsilon", "This defines epsilon for smoothening of sign().", 0.0 );
         config.addEntry        < double > ( "-value", "Constant value of RHS.", 0.0 );
      }
};

