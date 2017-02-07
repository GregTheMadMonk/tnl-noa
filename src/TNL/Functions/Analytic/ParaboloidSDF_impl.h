/***************************************************************************
                          ParaboloidSDFSDF_impl.h  -  description
                             -------------------
    begin                : Oct 13, 2014
    copyright            : (C) 2014 by Tomas Sobotik

 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once 

#include <TNL/Functions/Analytic/ParaboloidSDF.h>

namespace TNL {
   namespace Functions {
      namespace Analytic {

template< int dimensions, typename Real >
ParaboloidSDFBase< dimensions, Real >::ParaboloidSDFBase()
: xCenter( 0 ), yCenter( 0 ), zCenter( 0 ),
  coefficient( 1 ), radius ( 0 )
{
}

template< int dimensions, typename Real >
bool ParaboloidSDFBase< dimensions, Real >::setup( const Config::ParameterContainer& parameters,
        								 const String& prefix)
{
   this->xCenter = parameters.getParameter< double >( "x-center" );
   this->yCenter = parameters.getParameter< double >( "y-center" );
   this->zCenter = parameters.getParameter< double >( "z-center" );
   this->coefficient = parameters.getParameter< double >( "coefficient" );
   this->radius = parameters.getParameter< double >( "radius" );

   return true;
}

template< int dimensions, typename Real >
void ParaboloidSDFBase< dimensions, Real >::setXCenter( const Real& xCenter )
{
   this->xCenter = xCenter;
}

template< int dimensions, typename Real >
Real ParaboloidSDFBase< dimensions, Real >::getXCenter() const
{
   return this->xCenter;
}

template< int dimensions, typename Real >
void ParaboloidSDFBase< dimensions, Real >::setYCenter( const Real& yCenter )
{
   this->yCenter = yCenter;
}

template< int dimensions, typename Real >
Real ParaboloidSDFBase< dimensions, Real >::getYCenter() const
{
   return this->yCenter;
}
template< int dimensions, typename Real >
void ParaboloidSDFBase< dimensions, Real >::setZCenter( const Real& zCenter )
{
   this->zCenter = zCenter;
}

template< int dimensions, typename Real >
Real ParaboloidSDFBase< dimensions, Real >::getZCenter() const
{
   return this->zCenter;
}

template< int dimensions, typename Real >
void ParaboloidSDFBase< dimensions, Real >::setCoefficient( const Real& amplitude )
{
   this->coefficient = coefficient;
}

template< int dimensions, typename Real >
Real ParaboloidSDFBase< dimensions, Real >::getCoefficient() const
{
   return this->coefficient;
}

template< int dimensions, typename Real >
void ParaboloidSDFBase< dimensions, Real >::setOffset( const Real& offset )
{
   this->radius = offset;
}

template< int dimensions, typename Real >
Real ParaboloidSDFBase< dimensions, Real >::getOffset() const
{
   return this->radius;
}

template< typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder>
__cuda_callable__
Real
ParaboloidSDF< 1, Real >::
getPartialDerivative( const VertexType& v,
                      const Real& time ) const
{
   const Real& x = v.x();
   if( YDiffOrder != 0 || ZDiffOrder != 0 )
      return 0.0;
   if( XDiffOrder == 0 )
      return ::sqrt( ( x - this -> xCenter ) * ( x - this -> xCenter ) ) - this->radius;
   if( XDiffOrder == 1 )
      return 1.0;
   return 0.0;
}


template< typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder>
__cuda_callable__
Real
ParaboloidSDF< 2, Real >::
getPartialDerivative( const VertexType& v,
                      const Real& time ) const
{
   const Real& x = v.x();
   const Real& y = v.y();
   if( XDiffOrder == 0 && YDiffOrder == 0 && ZDiffOrder == 0 )
   {
      return ::sqrt ( ( x - this -> xCenter ) * ( x - this -> xCenter )
    		  	  + ( y - this -> yCenter ) * ( y - this -> yCenter ) ) - this->radius;
   }
   return 0.0;
}

template< typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder>
__cuda_callable__
Real
ParaboloidSDF< 3, Real >::
getPartialDerivative( const VertexType& v,
                      const Real& time ) const
{
   const Real& x = v.x();
   const Real& y = v.y();
   const Real& z = v.z();
   if( XDiffOrder == 0 && YDiffOrder == 0 && ZDiffOrder == 0 )
   {
      return ::sqrt( ( x - this -> xCenter ) * ( x - this -> xCenter )
    		  	 + ( y - this -> yCenter ) * ( y - this -> yCenter )
    		  	 + ( z - this -> zCenter ) * ( z - this -> zCenter ) ) - this->radius;
   }
   return 0.0;
}
         
      } //namespace Analytic
   } // namepsace Functions
} // namespace TNL
