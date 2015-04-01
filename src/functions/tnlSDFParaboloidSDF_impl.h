/***************************************************************************
                          tnlSDFParaboloidSDF_impl.h  -  description
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

template< typename Real >
tnlSDFParaboloidSDFBase< Real >::tnlSDFParaboloidSDFBase()
: xCentre( 0 ), yCentre( 0 ), zCentre( 0 ),
  coefficient( 1 ), offset ( 0 )
{
}

template< typename Real >
bool tnlSDFParaboloidSDFBase< Real >::setup( const tnlParameterContainer& parameters,
        									const tnlString& prefix )
{
   this->xCentre = parameters.getParameter< double >( "x-centre" );
   this->yCentre = parameters.getParameter< double >( "y-centre" );
   this->zCentre = parameters.getParameter< double >( "z-centre" );
   this->coefficient = parameters.getParameter< double >( "coefficient" );
   this->offset = parameters.getParameter< double >( "offset" );

   return true;
}

template< typename Real >
void tnlSDFParaboloidSDFBase< Real >::setXCentre( const Real& xCentre )
{
   this->xCentre = xCentre;
}

template< typename Real >
Real tnlSDFParaboloidSDFBase< Real >::getXCentre() const
{
   return this->xCentre;
}

template< typename Real >
void tnlSDFParaboloidSDFBase< Real >::setYCentre( const Real& yCentre )
{
   this->yCentre = yCentre;
}

template< typename Real >
Real tnlSDFParaboloidSDFBase< Real >::getYCentre() const
{
   return this->yCentre;
}
template< typename Real >
void tnlSDFParaboloidSDFBase< Real >::setZCentre( const Real& zCentre )
{
   this->zCentre = zCentre;
}

template< typename Real >
Real tnlSDFParaboloidSDFBase< Real >::getZCentre() const
{
   return this->zCentre;
}

template< typename Real >
void tnlSDFParaboloidSDFBase< Real >::setCoefficient( const Real& coefficient )
{
   this->coefficient = coefficient;
}

template< typename Real >
Real tnlSDFParaboloidSDFBase< Real >::getCoefficient() const
{
   return this->coefficient;
}

template< typename Real >
void tnlSDFParaboloidSDFBase< Real >::setOffset( const Real& offset )
{
   this->offset = offset;
}

template< typename Real >
Real tnlSDFParaboloidSDFBase< Real >::getOffset() const
{
   return this->offset;
}

template< typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder,
             typename Vertex >
Real tnlSDFParaboloidSDF< 1, Real >::getValue( const Vertex& v,
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
             int ZDiffOrder,
             typename Vertex >
Real tnlSDFParaboloidSDF< 2, Real >::getValue( const Vertex& v,
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
             int ZDiffOrder,
             typename Vertex >
Real tnlSDFParaboloidSDF< 3, Real >::getValue( const Vertex& v,
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
