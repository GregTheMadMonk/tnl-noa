/***************************************************************************
                          tnlConstantFunction_impl.h  -  description
                             -------------------
    begin                : Aug 2, 2014
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

#ifndef TNLCONSTANTFUNCTION_IMPL_H_
#define TNLCONSTANTFUNCTION_IMPL_H_

template< int Dimensions,
          typename Real >
tnlConstantFunction< Dimensions, Real >::
tnlConstantFunction()
: value( 0.0 )
{
}

template< int Dimensions,
          typename Real >
void
tnlConstantFunction< Dimensions, Real >::
setValue( const RealType& value )
{
   this->value = value;
}

template< int Dimensions,
          typename Real >
const Real&
tnlConstantFunction< Dimensions, Real >::
getValue() const
{
   return this->value;
}

template< int Dimensions,
          typename Real >
bool
tnlConstantFunction< Dimensions, Real >::
init( const tnlParameterContainer& parameters,
      const tnlString& prefix )
{
   this->setValue( parameters.GetParameter< double >( prefix + "-value") );
   return true;
}

template< int Dimensions,
          typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder >
Real
tnlConstantFunction< Dimensions, Real >::
getValue( const VertexType& v ) const
{
   return value;
}

#endif
