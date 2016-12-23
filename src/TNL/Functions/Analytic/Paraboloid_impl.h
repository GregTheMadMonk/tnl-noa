/***************************************************************************
                          Paraboloid_impl.h  -  description
                             -------------------
    begin                : Oct 13, 2014
    copyright            : (C) 2014 by Tomas Sobotik

 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once 

#include <TNL/Functions/Analytic/Paraboloid.h>

namespace TNL {
   namespace Functions {
      namespace Analytic {

template< int dimensions, typename Real >
ParaboloidBase< dimensions, Real >::ParaboloidBase()
: xCentre( 0 ), yCentre( 0 ), zCentre( 0 ),
  coefficient( 1 ), radius ( 0 )
{
}

template< int dimensions, typename Real >
bool ParaboloidBase< dimensions, Real >::setup( const Config::ParameterContainer& parameters,
        								 const String& prefix)
{
   this->xCentre = parameters.getParameter< double >( "x-centre" );
   this->yCentre = parameters.getParameter< double >( "y-centre" );
   this->zCentre = parameters.getParameter< double >( "z-centre" );
   this->coefficient = parameters.getParameter< double >( "coefficient" );
   this->radius = parameters.getParameter< double >( "radius" );

   return true;
}

template< int dimensions, typename Real >
void ParaboloidBase< dimensions, Real >::setXCentre( const Real& xCentre )
{
   this->xCentre = xCentre;
}

template< int dimensions, typename Real >
Real ParaboloidBase< dimensions, Real >::getXCentre() const
{
   return this->xCentre;
}

template< int dimensions, typename Real >
void ParaboloidBase< dimensions, Real >::setYCentre( const Real& yCentre )
{
   this->yCentre = yCentre;
}

template< int dimensions, typename Real >
Real ParaboloidBase< dimensions, Real >::getYCentre() const
{
   return this->yCentre;
}
template< int dimensions, typename Real >
void ParaboloidBase< dimensions, Real >::setZCentre( const Real& zCentre )
{
   this->zCentre = zCentre;
}

template< int dimensions, typename Real >
Real ParaboloidBase< dimensions, Real >::getZCentre() const
{
   return this->zCentre;
}

template< int dimensions, typename Real >
void ParaboloidBase< dimensions, Real >::setCoefficient( const Real& amplitude )
{
   this->coefficient = coefficient;
}

template< int dimensions, typename Real >
Real ParaboloidBase< dimensions, Real >::getCoefficient() const
{
   return this->coefficient;
}

template< int dimensions, typename Real >
void ParaboloidBase< dimensions, Real >::setOffset( const Real& offset )
{
   this->radius = offset;
}

template< int dimensions, typename Real >
Real ParaboloidBase< dimensions, Real >::getOffset() const
{
   return this->radius;
}

template< typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder>
__cuda_callable__
Real
Paraboloid< 1, Real >::
getPartialDerivative( const VertexType& v,
                      const Real& time ) const
{
   const Real& x = v.x();
   if( YDiffOrder != 0 || ZDiffOrder != 0 )
      return 0.0;
   if( XDiffOrder == 0 )
      return this->coefficient * ( ( x - this -> xCentre ) * ( x - this -> xCentre ) - this->radius*this->radius );
   if( XDiffOrder == 1 )
      return 2.0 * this->coefficient * ( x - this -> xCentre );
   return 0.0;
}


template< typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder>
__cuda_callable__
Real
Paraboloid< 2, Real >::
getPartialDerivative( const VertexType& v,
                      const Real& time ) const
{
   const Real& x = v.x();
   const Real& y = v.y();
   if( ZDiffOrder != 0 )
      return 0.0;
   if( XDiffOrder == 0 && YDiffOrder == 0 && ZDiffOrder == 0 )
   {
      return this->coefficient * ( ( x - this -> xCentre ) * ( x - this -> xCentre )
    		  	  	  	         + ( y - this -> yCentre ) * ( y - this -> yCentre ) - this->radius*this->radius );
   }
   if( XDiffOrder == 1 && YDiffOrder == 0)
	   return 2.0 * this->coefficient * ( x - this -> xCentre );
   if( YDiffOrder == 1 && XDiffOrder == 0)
	   return 2.0 * this->coefficient * ( y - this -> yCentre );
   if( XDiffOrder == 2 && YDiffOrder == 0)
	   return 2.0 * this->coefficient;
   if( YDiffOrder == 2 && XDiffOrder == 0)
	   return 2.0 * this->coefficient;
   return 0.0;
}

template< typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder>
__cuda_callable__
Real
Paraboloid< 3, Real >::
getPartialDerivative( const VertexType& v,
                      const Real& time ) const
{
   const Real& x = v.x();
   const Real& y = v.y();
   const Real& z = v.z();
   if( XDiffOrder == 0 && YDiffOrder == 0 && ZDiffOrder == 0 )
   {
      return this->coefficient * ( ( x - this -> xCentre ) * ( x - this -> xCentre )
    		  	  	  	         + ( y - this -> yCentre ) * ( y - this -> yCentre )
    		  	  	  	         + ( z - this -> zCentre ) * ( z - this -> zCentre ) - this->radius*this->radius );
   }
   if( XDiffOrder == 1 && YDiffOrder == 0 && ZDiffOrder == 0)
	   return 2.0 * this->coefficient * ( x - this -> xCentre );
   if( YDiffOrder == 1 && XDiffOrder == 0 && ZDiffOrder == 0)
	   return 2.0 * this->coefficient * ( y - this -> yCentre );
   if( ZDiffOrder == 1 && XDiffOrder == 0 && YDiffOrder == 0)
	   return 2.0 * this->coefficient * ( z - this -> zCentre );
   if( XDiffOrder == 2 && YDiffOrder == 0 && ZDiffOrder == 0)
	   return 2.0 * this->coefficient;
   if( YDiffOrder == 2 && XDiffOrder == 0 && ZDiffOrder == 0)
	   return 2.0 * this->coefficient;
   if( ZDiffOrder == 2 && XDiffOrder == 0 && YDiffOrder == 0)
	   return 2.0 * this->coefficient;
   return 0.0;
}
         
      } // namespace Analytic
   } // namedspace Functions
} // namespace TNL
