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
: xCentre( 0 ), yCentre( 0 ), zCentre( 0 ),
  coefficient( 1 ), radius ( 0 )
{
}

template< int dimensions, typename Real >
bool ParaboloidSDFBase< dimensions, Real >::setup( const Config::ParameterContainer& parameters,
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
void ParaboloidSDFBase< dimensions, Real >::setXCentre( const Real& xCentre )
{
   this->xCentre = xCentre;
}

template< int dimensions, typename Real >
Real ParaboloidSDFBase< dimensions, Real >::getXCentre() const
{
   return this->xCentre;
}

template< int dimensions, typename Real >
void ParaboloidSDFBase< dimensions, Real >::setYCentre( const Real& yCentre )
{
   this->yCentre = yCentre;
}

template< int dimensions, typename Real >
Real ParaboloidSDFBase< dimensions, Real >::getYCentre() const
{
   return this->yCentre;
}
template< int dimensions, typename Real >
void ParaboloidSDFBase< dimensions, Real >::setZCentre( const Real& zCentre )
{
   this->zCentre = zCentre;
}

template< int dimensions, typename Real >
Real ParaboloidSDFBase< dimensions, Real >::getZCentre() const
{
   return this->zCentre;
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
      return ::sqrt( ( x - this -> xCentre ) * ( x - this -> xCentre ) ) - this->radius;
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
      return ::sqrt ( ( x - this -> xCentre ) * ( x - this -> xCentre )
    		  	  + ( y - this -> yCentre ) * ( y - this -> yCentre ) ) - this->radius;
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
      return ::sqrt( ( x - this -> xCentre ) * ( x - this -> xCentre )
    		  	 + ( y - this -> yCentre ) * ( y - this -> yCentre )
    		  	 + ( z - this -> zCentre ) * ( z - this -> zCentre ) ) - this->radius;
   }
   return 0.0;
}
         
      } //namespace Analytic
   } // namepsace Functions
} // namespace TNL
