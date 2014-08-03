/***************************************************************************
                          tnlTestFunction_impl.h  -  description
                             -------------------
    begin                : Aug 3, 2014
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

#ifndef TNLTESTFUNCTION_IMPL_H_
#define TNLTESTFUNCTION_IMPL_H_

#include <functions/tnlConstantFunction.h>
#include <functions/tnlExpBumpFunction.h>
#include <functions/tnlSinBumpsFunction.h>
#include <functions/tnlSinWavesFunction.h>

template< int FunctionDimensions,
          typename Real,
          typename Device >
tnlTestingFunction< FunctionDimensions, Real, Device >::
tnlTestingFunction()
: function( 0 ),
  functionType( none )
{
}

template< int FunctionDimensions,
          typename Real,
          typename Device >
bool
tnlTestingFunction< FunctionDimensions, Real, Device >::
init( const tnlParameterContainer& parameters )
{
   const tnlString& testFunction = parameters.getParameter< tnlString >( "test-function" );

   if( testFunction == "constant" )
   {
      typedef tnlConstantFunction< Dimensions, Real > FunctionType;
      FunctionType* auxFunction = new FunctionType;
      if( ! auxFunction->init( parameters ) )
      {
         delete auxFunction;
         return false;
      }
      functionType = constant;
      if( Device::DeviceType == tnlHostType )
      {
         function = auxFunction;
      }
      if( Device::DeviceType == tnlCudaType )
      {
         function = passToDevice( *auxFunction );
      }
   }
   if( testFunction == "exp-bump" )
   {

   }
   if( testFunction == "sin-bumps" )
   {

   }
   if( testFunction == "sin-waves" )
   {

   }
}

template< int FunctionDimensions,
          typename Real,
          typename Device >
tnlTestingFunction< FunctionDimensions, Real, Device >::
~tnlTestingFunction()
{
   if( Device::DeviceType == tnlHostType )
   {
      switch( functionType )
      {
         case constant:
            delete ( tnlConstantFunction< Dimensions, Real> * ) function;
            break;
      }
   }

}


#endif /* TNLTESTFUNCTION_IMPL_H_ */
