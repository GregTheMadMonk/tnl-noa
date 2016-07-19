/***************************************************************************
                          tnlExpBumpFunction_impl.h  -  description
                             -------------------
    begin                : Dec 5, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/functions/initial_conditions/level_set_functions/tnlPseudoSquareFunction.h>

namespace TNL {

template< typename Real,
          int Dimensions >
bool
tnlPseudoSquareFunctionBase< Real, Dimensions >::
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
tnlPseudoSquareFunction< 1, Real >::getType()
{
   return "tnlPseudoSquareFunction< 1, " + TNL::getType< Real >() + tnlString( " >" );
}

template< typename Real >
tnlPseudoSquareFunction< 1, Real >::tnlPseudoSquareFunction()
{
}

template< typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder >
__cuda_callable__
Real
tnlPseudoSquareFunction< 1, Real >::
getPartialDerivative( const VertexType& v,
                      const Real& time ) const
{
   const RealType& x = v.x();
   if( YDiffOrder != 0 || ZDiffOrder != 0 )
      return 0.0;
   if( XDiffOrder == 0 )
      return 0.0;
   return 0.0;
}

template< typename Real >
__cuda_callable__
Real
tnlPseudoSquareFunction< 1, Real >::
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
tnlPseudoSquareFunction< 2, Real >::getType()
{
   return tnlString( "tnlPseudoSquareFunction< 2, " ) + TNL::getType< Real >() + " >";
}

template< typename Real >
tnlPseudoSquareFunction< 2, Real >::tnlPseudoSquareFunction()
{
}

template< typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder >
__cuda_callable__
Real
tnlPseudoSquareFunction< 2, Real >::
getPartialDerivative( const VertexType& v,
                      const Real& time ) const
{
   const RealType& x = v.x();
   const RealType& y = v.y();
   if( ZDiffOrder != 0 )
      return 0.0;
   if( XDiffOrder == 0 && YDiffOrder == 0 )
      return x * x + y * y - this->height - ::cos( 2 * x * y ) * ::cos( 2 * x * y );
   return 0.0;
}

template< typename Real >
__cuda_callable__
Real
tnlPseudoSquareFunction< 2, Real >::
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
tnlPseudoSquareFunction< 3, Real >::getType()
{
   return tnlString( "tnlPseudoSquareFunction< 3, " ) + TNL::getType< Real >() + " >";
}

template< typename Real >
tnlPseudoSquareFunction< 3, Real >::tnlPseudoSquareFunction()
{
}

template< typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder >
__cuda_callable__
Real
tnlPseudoSquareFunction< 3, Real >::
getPartialDerivative( const VertexType& v,
                      const Real& time ) const
{
   const RealType& x = v.x();
   const RealType& y = v.y();
   const RealType& z = v.z();
   if( XDiffOrder == 0 && YDiffOrder == 0 && ZDiffOrder == 0 )
      return 0.0;
   return 0.0;
}

template< typename Real >
__cuda_callable__
Real
tnlPseudoSquareFunction< 3, Real >::
operator()( const VertexType& v,
            const Real& time ) const
{
   return this->template getPartialDerivative< 0, 0, 0 >( v, time );
}

} // namepsace TNL
