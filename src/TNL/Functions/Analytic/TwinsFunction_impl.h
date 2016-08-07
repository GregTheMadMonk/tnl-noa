/***************************************************************************
                          ExpBumpFunction_impl.h  -  description
                             -------------------
    begin                : Dec 5, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Functions/Analytic/TwinsFunction.h>

namespace TNL {
namespace Functions {
namespace Analytic {   

template< typename Real,
          int Dimensions >
bool
TwinsFunctionBase< Real, Dimensions >::
setup( const Config::ParameterContainer& parameters,
       const String& prefix )
{
   return true;
}


/***
 * 1D
 */

template< typename Real >
String
TwinsFunction< 1, Real >::getType()
{
   return "TwinsFunction< 1, " + TNL::getType< Real >() + String( " >" );
}

template< typename Real >
TwinsFunction< 1, Real >::TwinsFunction()
{
}

template< typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder,
             typename Vertex >
__cuda_callable__
Real
TwinsFunction< 1, Real >::getPartialDerivative( const Vertex& v,
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
TwinsFunction< 1, Real >::
operator()( const VertexType& v,
            const Real& time ) const
{
   return this->template getPartialDerivative< 0, 0, 0 >( v, time );
}


/****
 * 2D
 */
template< typename Real >
String
TwinsFunction< 2, Real >::getType()
{
   return String( "TwinsFunction< 2, " ) + TNL::getType< Real >() + " >";
}

template< typename Real >
TwinsFunction< 2, Real >::TwinsFunction()
{
}

template< typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder,
             typename Vertex >
__cuda_callable__
Real
TwinsFunction< 2, Real >::
getPartialDerivative( const Vertex& v,
                      const Real& time ) const
{
   const RealType& x = v.x();
   const RealType& y = v.y();
   if( ZDiffOrder != 0 )
      return 0.0;
   if( XDiffOrder == 0 && YDiffOrder == 0 )
      return -0.5 * ::sin( M_PI * x) * ::sin( M_PI * x) * ( 1 - ( y - 2 ) * ( y - 2 ) ) * ( 1 - ::tanh( 10 * ( ::sqrt( x * x + y * y ) - 0.6 ) ) );
   return 0.0;
}

template< typename Real >
__cuda_callable__
Real
TwinsFunction< 2, Real >::
operator()( const VertexType& v,
            const Real& time ) const
{
   return this->template getPartialDerivative< 0, 0, 0 >( v, time );
}


/****
 * 3D
 */
template< typename Real >
String
TwinsFunction< 3, Real >::getType()
{
   return String( "TwinsFunction< 3, " ) + TNL::getType< Real >() + " >";
}

template< typename Real >
TwinsFunction< 3, Real >::TwinsFunction()
{
}

template< typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder,
             typename Vertex >
__cuda_callable__
Real
TwinsFunction< 3, Real >::
getPartialDerivative( const Vertex& v,
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
TwinsFunction< 3, Real >::
operator()( const VertexType& v,
            const Real& time ) const
{
   return this->template getPartialDerivative< 0, 0, 0 >( v, time );
}

} // namespace Analytic
} // namespace Functions
} // namespace TNL

