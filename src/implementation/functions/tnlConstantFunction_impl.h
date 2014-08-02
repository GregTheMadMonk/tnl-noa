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
          typename Vertex,
          typename Device >
tnlConstantFunction< Dimensions, Vertex, Device >::
tnlConstantFunction()
: value( 0.0 )
{
}

template< int Dimensions,
          typename Vertex,
          typename Device >
void
tnlConstantFunction< Dimensions, Vertex, Device >::
setValue( const RealType& value )
{
   this->value = value;
}

template< int Dimensions,
          typename Vertex,
          typename Device >
const typename Vertex::RealType&
tnlConstantFunction< Dimensions, Vertex, Device >::
getValue() const
{
   return this->value;
}

template< int Dimensions,
          typename Vertex,
          typename Device >
bool
tnlConstantFunction< Dimensions, Vertex, Device >::
init( const tnlParameterContainer& parameters )
{

}

template< int Dimensions,
          typename Vertex,
          typename Device >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder >
typename Vertex::RealType
tnlConstantFunction< Dimensions, Vertex, Device >::
getValue( const VertexType& v ) const
{
   return value;
}

#endif
