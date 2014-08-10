/***************************************************************************
                          heatEquationConfig.h  -  description
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

#ifndef HEATEQUATIONCONFIG_H_
#define HEATEQUATIONCONFIG_H_

#include <config/tnlConfigDescription.h>
#include <functions/tnlTestFunction.h>

template< typename ConfigTag >
class heatEquationConfig
{
   public:
      static void configSetup( tnlConfigDescription& config )
      {
         config.addDelimiter( "Heat equation settings:" );
         config.addDelimiter( "Tests setting::" );
         config.addEntry     < bool >( "approximation-test", "Test of the Laplace operator approximation.", false );
         config.addEntry     < bool >( "eoc-test", "Test of the numerical scheme convergence.", false );
         tnlTestFunction< 3, double >::configSetup( config );
      }
};

#endif /* HEATEQUATIONCONFIG_H_ */
