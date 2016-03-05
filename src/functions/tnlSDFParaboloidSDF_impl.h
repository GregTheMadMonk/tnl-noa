/***************************************************************************
                          tnlSDFParaboloidSDFSDF_impl.h  -  description
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

#ifndef TNLSDFPARABOLOIDSDF_IMPL_H_
#define TNLSDFPARABOLOIDSDF_IMPL_H_

#include <functions/tnlSDFParaboloidSDF.h>

template< int dimensions, typename Real >
tnlSDFParaboloidSDFBase< dimensions, Real >::tnlSDFParaboloidSDFBase()
: xCentre( 0 ), yCentre( 0 ), zCentre( 0 ),
  coefficient( 1 ), offset ( 0 )
{
}

template< int dimensions, typename Real >
bool tnlSDFParaboloidSDFBase< dimensions, Real >::setup( const tnlParameterContainer& parameters,
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
void tnlSDFParaboloidSDFBase< dimensions, Real >::setXCentre( const Real& xCentre )
{
   this->xCentre = xCentre;
}

template< int dimensions, typename Real >
Real tnlSDFParaboloidSDFBase< dimensions, Real >::getXCentre() const
{
   return this->xCentre;
}

template< int dimensions, typename Real >
void tnlSDFParaboloidSDFBase< dimensions, Real >::setYCentre( const Real& yCentre )
{
   this->yCentre = yCentre;
}

template< int dimensions, typename Real >
Real tnlSDFParaboloidSDFBase< dimensions, Real >::getYCentre() const
{
   return this->yCentre;
}
template< int dimensions, typename Real >
void tnlSDFParaboloidSDFBase< dimensions, Real >::setZCentre( const Real& zCentre )
{
   this->zCentre = zCentre;
}

template< int dimensions, typename Real >
Real tnlSDFParaboloidSDFBase< dimensions, Real >::getZCentre() const
{
   return this->zCentre;
}

template< int dimensions, typename Real >
void tnlSDFParaboloidSDFBase< dimensions, Real >::setCoefficient( const Real& amplitude )
{
   this->coefficient = coefficient;
}

template< int dimensions, typename Real >
Real tnlSDFParaboloidSDFBase< dimensions, Real >::getCoefficient() const
{
   return this->coefficient;
}

template< int dimensions, typename Real >
void tnlSDFParaboloidSDFBase< dimensions, Real >::setOffset( const Real& offset )
{
   this->offset = offset;
}

template< int dimensions, typename Real >
Real tnlSDFParaboloidSDFBase< dimensions, Real >::getOffset() const
{
   return this->offset;
}

template< typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder>
__cuda_callable__
Real
tnlSDFParaboloidSDF< 1, Real >::
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
tnlSDFParaboloidSDF< 2, Real >::
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
tnlSDFParaboloidSDF< 3, Real >::
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

#endif /* TNLSDFPARABOLOIDSDF_IMPL_H_ */
