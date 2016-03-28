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

#ifndef TNLFLOWERPOTFUNCTION_IMPL_H_
#define TNLFLOWERPOTFUNCTION_IMPL_H_

#include <functions/initial_conditions/tnlFlowerpotFunction.h>

template< typename Real,
          int Dimensions >
bool
tnlFlowerpotFunctionBase< Real, Dimensions >::
setup( const tnlParameterContainer& parameters,
       const tnlString& prefix )
{
   this->diameter = parameters.getParameter< double >( prefix + "diameter" );
   return true;
}

template< typename Real, 
          int Dimensions >
void tnlFlowerpotFunctionBase< Real, Dimensions >::setDiameter( const Real& sigma )
{
   this->diameter = diameter;
}

template< typename Real,
          int Dimensions >
const Real& tnlFlowerpotFunctionBase< Real, Dimensions >::getDiameter() const
{
   return this->diameter;
}

/***
 * 1D
 */

template< typename Real >
tnlString
tnlFlowerpotFunction< 1, Real >::getType()
{
   return "tnlFlowerpotFunction< 1, " + ::getType< Real >() + tnlString( " >" );
}

template< typename Real >
tnlFlowerpotFunction< 1, Real >::tnlFlowerpotFunction()
{
}

template< typename Real >
   template< int XDiffOrder, 
             int YDiffOrder,
             int ZDiffOrder,
             typename Vertex >
__cuda_callable__
Real
tnlFlowerpotFunction< 1, Real >::getPartialDerivative( const Vertex& v,
                                                       const Real& time ) const
{
   const RealType& x = v.x();
   if( YDiffOrder != 0 || ZDiffOrder != 0 )
      return 0.0;
   if( XDiffOrder == 0 )
      return sin( M_PI * tanh( 5 * ( x * x - this->diameter ) ) );
   return 0.0;
}

template< typename Real >
__cuda_callable__
Real
tnlFlowerpotFunction< 1, Real >::
operator()( const VertexType& v,
            const Real& time ) const
{
   return this->template getPartialDerivative< 0, 0, 0 >( v, time );
}


/****
 * 2D
 */
template< typename Real >
tnlString
tnlFlowerpotFunction< 2, Real >::getType()
{
   return tnlString( "tnlFlowerpotFunction< 2, " ) + ::getType< Real >() + " >";
}

template< typename Real >
tnlFlowerpotFunction< 2, Real >::tnlFlowerpotFunction()
{
}

template< typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder,
             typename Vertex >
__cuda_callable__
Real
tnlFlowerpotFunction< 2, Real >::
getPartialDerivative( const Vertex& v,
                      const Real& time ) const
{
   const RealType& x = v.x();
   const RealType& y = v.y();
   if( ZDiffOrder != 0 )
      return 0.0;
   if( XDiffOrder == 0 && YDiffOrder == 0 )
      return sin( M_PI * tanh( 5 * ( x * x + y * y - this->diameter ) ) );
   return 0.0;
}

template< typename Real >
__cuda_callable__
Real
tnlFlowerpotFunction< 2, Real >::
operator()( const VertexType& v,
            const Real& time ) const
{
   return this->template getPartialDerivative< 0, 0, 0 >( v, time );
}


/****
 * 3D
 */

template< typename Real >
tnlString
tnlFlowerpotFunction< 3, Real >::getType()
{
   return tnlString( "tnlFlowerpotFunction< 3, " ) + ::getType< Real >() + " >";
}

template< typename Real >
tnlFlowerpotFunction< 3, Real >::tnlFlowerpotFunction()
{
}

template< typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder,
             typename Vertex >
__cuda_callable__
Real
tnlFlowerpotFunction< 3, Real >::
getPartialDerivative( const Vertex& v,
                      const Real& time ) const
{
   const RealType& x = v.x();
   const RealType& y = v.y();
   const RealType& z = v.z();
   if( XDiffOrder == 0 && YDiffOrder == 0 && ZDiffOrder == 0 )
      return sin( M_PI * tanh( 5 * ( x * x + y * y + z * z - 0.25 ) ) );
   return 0.0;
}

template< typename Real >
__cuda_callable__
Real
tnlFlowerpotFunction< 3, Real >::
operator()( const VertexType& v,
            const Real& time ) const
{
   return this->template getPartialDerivative< 0, 0, 0 >( v, time );
}

#endif /* TNLFLOWERPOTFUNCTION_IMPL_H_ */