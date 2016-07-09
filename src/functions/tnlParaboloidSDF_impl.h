/***************************************************************************
                          tnlParaboloidSDFSDF_impl.h  -  description
                             -------------------
    begin                : Oct 13, 2014
    copyright            : (C) 2014 by Tomas Sobotik

 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#pragma once 

#include <functions/tnlParaboloidSDF.h>

template< int dimensions, typename Real >
tnlParaboloidSDFBase< dimensions, Real >::tnlParaboloidSDFBase()
: xCentre( 0 ), yCentre( 0 ), zCentre( 0 ),
  coefficient( 1 ), offset ( 0 )
{
}

template< int dimensions, typename Real >
bool tnlParaboloidSDFBase< dimensions, Real >::setup( const tnlParameterContainer& parameters,
        								 const tnlString& prefix)
{
   this->xCentre = parameters.getParameter< double >( "x-centre" );
   this->yCentre = parameters.getParameter< double >( "y-centre" );
   this->zCentre = parameters.getParameter< double >( "z-centre" );
   this->coefficient = parameters.getParameter< double >( "coefficient" );
   this->offset = parameters.getParameter< double >( "offset" );

   return true;
}

template< int dimensions, typename Real >
void tnlParaboloidSDFBase< dimensions, Real >::setXCentre( const Real& xCentre )
{
   this->xCentre = xCentre;
}

template< int dimensions, typename Real >
Real tnlParaboloidSDFBase< dimensions, Real >::getXCentre() const
{
   return this->xCentre;
}

template< int dimensions, typename Real >
void tnlParaboloidSDFBase< dimensions, Real >::setYCentre( const Real& yCentre )
{
   this->yCentre = yCentre;
}

template< int dimensions, typename Real >
Real tnlParaboloidSDFBase< dimensions, Real >::getYCentre() const
{
   return this->yCentre;
}
template< int dimensions, typename Real >
void tnlParaboloidSDFBase< dimensions, Real >::setZCentre( const Real& zCentre )
{
   this->zCentre = zCentre;
}

template< int dimensions, typename Real >
Real tnlParaboloidSDFBase< dimensions, Real >::getZCentre() const
{
   return this->zCentre;
}

template< int dimensions, typename Real >
void tnlParaboloidSDFBase< dimensions, Real >::setCoefficient( const Real& amplitude )
{
   this->coefficient = coefficient;
}

template< int dimensions, typename Real >
Real tnlParaboloidSDFBase< dimensions, Real >::getCoefficient() const
{
   return this->coefficient;
}

template< int dimensions, typename Real >
void tnlParaboloidSDFBase< dimensions, Real >::setOffset( const Real& offset )
{
   this->offset = offset;
}

template< int dimensions, typename Real >
Real tnlParaboloidSDFBase< dimensions, Real >::getOffset() const
{
   return this->offset;
}

template< typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder>
__cuda_callable__
Real
tnlParaboloidSDF< 1, Real >::
getPartialDerivative( const VertexType& v,
                      const Real& time ) const
{
   const Real& x = v.x();
   if( YDiffOrder != 0 || ZDiffOrder != 0 )
      return 0.0;
   if( XDiffOrder == 0 )
      return sqrt( ( x - this -> xCentre ) * ( x - this -> xCentre ) ) - this->offset;
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
tnlParaboloidSDF< 2, Real >::
getPartialDerivative( const VertexType& v,
                      const Real& time ) const
{
   const Real& x = v.x();
   const Real& y = v.y();
   if( XDiffOrder == 0 && YDiffOrder == 0 && ZDiffOrder == 0 )
   {
      return sqrt ( ( x - this -> xCentre ) * ( x - this -> xCentre )
    		  	  + ( y - this -> yCentre ) * ( y - this -> yCentre ) ) - this->offset;
   }
   return 0.0;
}

template< typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder>
__cuda_callable__
Real
tnlParaboloidSDF< 3, Real >::
getPartialDerivative( const VertexType& v,
                      const Real& time ) const
{
   const Real& x = v.x();
   const Real& y = v.y();
   const Real& z = v.z();
   if( XDiffOrder == 0 && YDiffOrder == 0 && ZDiffOrder == 0 )
   {
      return sqrt( ( x - this -> xCentre ) * ( x - this -> xCentre )
    		  	 + ( y - this -> yCentre ) * ( y - this -> yCentre )
    		  	 + ( z - this -> zCentre ) * ( z - this -> zCentre ) ) - this->offset;
   }
   return 0.0;
}
