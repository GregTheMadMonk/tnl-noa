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
         config.addEntry     < tnlString >( "test-function", "Testing function.", "sin-wave" );
            config.addEntryEnum( "sin-wave" );
            config.addEntryEnum( "sin-bumps" );
            config.addEntryEnum( "exp-bump" );
         config.addEntry     < double >( "wave-length", "Wave length of the sine based test functions.", 1.0 );
         config.addEntry     < double >( "wave-length-x", "Wave length of the sine based test functions.", 1.0 );
         config.addEntry     < double >( "wave-length-y", "Wave length of the sine based test functions.", 1.0 );
         config.addEntry     < double >( "wave-length-z", "Wave length of the sine based test functions.", 1.0 );
         config.addEntry     < double >( "phase", "Phase of the sine based test functions.", 0.0 );
         config.addEntry     < double >( "phase-x", "Phase of the sine based test functions.", 0.0 );
         config.addEntry     < double >( "phase-y", "Phase of the sine based test functions.", 0.0 );
         config.addEntry     < double >( "phase-z", "Phase of the sine based test functions.", 0.0 );
         config.addEntry     < double >( "amplitude", "Amplitude length of the sine based test functions.", 1.0 );
         config.addEntry     < double >( "waves-number", "Cut-off for the sine based test functions.", 0.0 );
         config.addEntry     < double >( "waves-number-x", "Cut-off for the sine based test functions.", 0.0 );
         config.addEntry     < double >( "waves-number-y", "Cut-off for the sine based test functions.", 0.0 );
         config.addEntry     < double >( "waves-number-z", "Cut-off for the sine based test functions.", 0.0 );
         config.addEntry     < double >( "sigma", "Sigma for the exp based test functions.", 1.0 );
         config.addEntry     < tnlString >( "test-function-time-dependence", "Time dependence of the test function.", "none" );
            config.addEntryEnum( "none" );
            config.addEntryEnum( "linear" );
            config.addEntryEnum( "quadratic" );
            config.addEntryEnum( "cosine" );
      }
};

#endif /* HEATEQUATIONCONFIG_H_ */
