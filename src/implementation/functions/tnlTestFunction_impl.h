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
#include <functions/tnlSinWaveFunction.h>

template< int FunctionDimensions,
          typename Real,
          typename Device >
tnlTestFunction< FunctionDimensions, Real, Device >::
tnlTestFunction()
: function( 0 ),
  functionType( none )
{
}

template< int FunctionDimensions,
          typename Real,
          typename Device >
   template< typename FunctionType >
bool
tnlTestFunction< FunctionDimensions, Real, Device >::
initFunction( const tnlParameterContainer& parameters )
{
   FunctionType* auxFunction = new FunctionType;
   if( ! auxFunction->init( parameters ) )
   {
      delete auxFunction;
      return false;
   }

   if( Device::DeviceType == tnlHostDevice )
   {
      function = auxFunction;
   }
   if( Device::DeviceType == tnlCudaDevice )
   {
      function = passToDevice( *auxFunction );
      delete auxFunction;
   }
}

template< int FunctionDimensions,
          typename Real,
          typename Device >
bool
tnlTestFunction< FunctionDimensions, Real, Device >::
init( const tnlParameterContainer& parameters )
{
   const tnlString& testFunction = parameters.GetParameter< tnlString >( "test-function" );

   if( testFunction == "constant" )
   {
      typedef tnlConstantFunction< Dimensions, Real > FunctionType;
      functionType = constant;
      return initFunction< FunctionType >( parameters );
   }
   if( testFunction == "exp-bump" )
   {
      typedef tnlExpBumpFunction< Dimensions, Real > FunctionType;
      functionType = expBump;
      return initFunction< FunctionType >( parameters );
   }
   if( testFunction == "sin-bumps" )
   {
      typedef tnlSinBumpsFunction< Dimensions, Real > FunctionType;
      functionType = sinBumps;
      return initFunction< FunctionType >( parameters );
   }
   if( testFunction == "sin-wave" )
   {
      typedef tnlSinWaveFunction< Dimensions, Real > FunctionType;
      functionType = sinWave;
      return initFunction< FunctionType >( parameters );
   }
}

template< int FunctionDimensions,
          typename Real,
          typename Device >
   template< typename Vertex >
#ifdef HAVE_CUDA
   __device__ __host__
#endif
Real
tnlTestFunction< FunctionDimensions, Real, Device >::
getValue( const Vertex& vertex ) const
{
   switch( functionType )
   {
      case constant:
         return ( ( tnlConstantFunction< Dimensions, Real >* ) function )->getValue( vertex );
         break;
      case expBump:
         return ( ( tnlExpBumpFunction< Dimensions, Real >* ) function )->getValue( vertex );
         break;
      case sinBumps:
         return ( ( tnlSinBumpsFunction< Dimensions, Real >* ) function )->getValue( vertex );
         break;
      case sinWave:
         return ( ( tnlSinWaveFunction< Dimensions, Real >* ) function )->getValue( vertex );
         break;
      default:
         return 0.0;
         break;
   }
}

template< int FunctionDimensions,
          typename Real,
          typename Device >
   template< typename FunctionType >
void
tnlTestFunction< FunctionDimensions, Real, Device >::
deleteFunction()
{
   if( Device::DeviceType == tnlHostDevice )
      delete ( FunctionType * ) function;
   if( Device::DeviceType == tnlCudaDevice )
      tnlCuda::freeFromDevice( ( FunctionType * ) function );
}

template< int FunctionDimensions,
          typename Real,
          typename Device >
tnlTestFunction< FunctionDimensions, Real, Device >::
~tnlTestFunction()
{
   switch( functionType )
   {
      case constant:
         deleteFunction< tnlConstantFunction< Dimensions, Real> >();
         break;
      case expBump:
         deleteFunction< tnlExpBumpFunction< Dimensions, Real> >();
         break;
      case sinBumps:
         deleteFunction< tnlSinBumpsFunction< Dimensions, Real> >();
         break;
      case sinWave:
         deleteFunction< tnlSinWaveFunction< Dimensions, Real> >();
         break;


   }


}


#endif /* TNLTESTFUNCTION_IMPL_H_ */
