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

#include <TNL/Functions/Analytic/ExpBump.h>

using namespace TNL;

template< typename Function >
class tnlPDEOperatorEocTestFunctionSetter
{
};

template< int Dimension,
          typename Real >
class tnlPDEOperatorEocTestFunctionSetter< Functions::Analytic::ExpBump< Dimension, Real > >
{
   static_assert( Dimension >= 0 && Dimension <= 3,
      "Wrong parameter Dimension." );
   public:
 
      typedef Functions::Analytic::ExpBump< Dimension, Real > FunctionType;
 
      static void setup( FunctionType& function )
      {
         function.setAmplitude( 1.5 );
         function.setSigma( 0.5 );
      }
};

#endif	/* TNLPDEOPERATOREOCTESTFUNCTIONSETTER_H */

