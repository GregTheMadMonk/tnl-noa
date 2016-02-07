/***************************************************************************
                          tnlPDEOperatorEocTestFunctionSetter.h  -  description
                             -------------------
    begin                : Feb 1, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
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

#ifndef TNLPDEOPERATOREOCTESTFUNCTIONSETTER_H
#define	TNLPDEOPERATOREOCTESTFUNCTIONSETTER_H

#include<functions/tnlExpBumpFunction.h>

template< typename Function >
class tnlPDEOperatorEocTestFunctionSetter
{
};

template< int Dimensions,
          typename Real >
class tnlPDEOperatorEocTestFunctionSetter< tnlExpBumpFunction< Dimensions, Real > >
{
   static_assert( Dimensions >= 0 && Dimensions <= 3,
      "Wrong parameter Dimensions." );
   public:      
      
      typedef tnlExpBumpFunction< Dimensions, Real > FunctionType;
      
      static void setup( FunctionType& function )
      {
         function.setAmplitude( 1.5 );
         function.setSigma( 0.5 );
      }
};

#endif	/* TNLPDEOPERATOREOCTESTFUNCTIONSETTER_H */

