/***************************************************************************
                          tnlExpBumpFunction_impl.h  -  description
                             -------------------
    begin                : Dec 5, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
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

#ifndef TNLBLOBFUNCTION_IMPL_H_
#define TNLBLOBFUNCTION_IMPL_H_

#include <functions/initial_conditions/level_set_functions/tnlBlobFunction.h>

template< typename Real,
          int Dimensions >
bool
tnlBlobFunctionBase< Real, Dimensions >::
setup( const tnlParameterContainer& parameters,
       const tnlString& prefix )
{
   this->height = parameters.getParameter< double >( prefix + "height" );
   
   return true;
}

/***
 * 1D
 */

template< typename Real >
tnlString
tnlBlobFunction< 1, Real >::getType()
{
   return "tnlBlobFunction< 1, " + ::getType< Real >() + tnlString( " >" );
}

template< typename Real >
tnlBlobFunction< 1, Real >::tnlBlobFunction()
{
}

template< typename Real >
   template< int XDiffOrder, 
             int YDiffOrder,
             int ZDiffOrder,
             typename Vertex >
__cuda_callable__
Real
tnlBlobFunction< 1, Real >::getValue( const Vertex& v,
                                         const Real& time ) const
{
   const RealType& x = v.x();
   if( YDiffOrder != 0 || ZDiffOrder != 0 )
      return 0.0;
   if( XDiffOrder == 0 )
      return 0.0;
   return 0.0;
}

/****
 * 2D
 */

template< typename Real >
tnlString
tnlBlobFunction< 2, Real >::getType()
{
   return tnlString( "tnlBlobFunction< 2, " ) + ::getType< Real >() + " >";
}

template< typename Real >
tnlBlobFunction< 2, Real >::tnlBlobFunction()
{
}

template< typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder,
             typename Vertex >
__cuda_callable__
Real
tnlBlobFunction< 2, Real >::
getValue( const Vertex& v,
          const Real& time ) const
{
   const RealType& x = v.x();
   const RealType& y = v.y();
   if( ZDiffOrder != 0 )
      return 0.0;
   if( XDiffOrder == 0 && YDiffOrder == 0 )
      return x * x + y * y - this->height - sin( cos ( 2 * x + y ) * sin ( 2 * x + y ) );
   return 0.0;
}

/****
 * 3D
 */

template< typename Real >
tnlString
tnlBlobFunction< 3, Real >::getType()
{
   return tnlString( "tnlBlobFunction< 3, " ) + ::getType< Real >() + " >";
}

template< typename Real >
tnlBlobFunction< 3, Real >::tnlBlobFunction()
{
}

template< typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder,
             typename Vertex >
__cuda_callable__
Real
tnlBlobFunction< 3, Real >::
getValue( const Vertex& v,
          const Real& time ) const
{
   const RealType& x = v.x();
   const RealType& y = v.y();
   const RealType& z = v.z();
   if( XDiffOrder == 0 && YDiffOrder == 0 && ZDiffOrder == 0 )
      return 0.0;
   return 0.0;
}

#endif /* TNLBLOBFUNCTION_IMPL_H_ */
