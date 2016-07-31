/***************************************************************************
                          tnlPDEOperatorEocTestFunctionSetter.h  -  description
                             -------------------
    begin                : Feb 1, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLPDEOPERATOREOCTESTFUNCTIONSETTER_H
#define	TNLPDEOPERATOREOCTESTFUNCTIONSETTER_H

#include <TNL/Functions/Analytic/tnlExpBumpFunction.h>

using namespace TNL;

template< typename Function >
class tnlPDEOperatorEocTestFunctionSetter
{
};

template< int Dimensions,
          typename Real >
class tnlPDEOperatorEocTestFunctionSetter< Functions::tnlExpBumpFunction< Dimensions, Real > >
{
   static_assert( Dimensions >= 0 && Dimensions <= 3,
      "Wrong parameter Dimensions." );
   public:
 
      typedef Functions::tnlExpBumpFunction< Dimensions, Real > FunctionType;
 
      static void setup( FunctionType& function )
      {
         function.setAmplitude( 1.5 );
         function.setSigma( 0.5 );
      }
};

#endif	/* TNLPDEOPERATOREOCTESTFUNCTIONSETTER_H */

