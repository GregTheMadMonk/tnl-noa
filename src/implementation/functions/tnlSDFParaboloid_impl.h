/***************************************************************************
                          tnlSDFParaboloid_impl.h  -  description
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

#ifndef TNLSDFPARABOLOID_IMPL_H_
#define TNLSDFPARABOLOID_IMPL_H_

#include <functions/tnlSDFParaboloid.h>

template< typename Real >
tnlSDFParaboloidBase< Real >::tnlSDFParaboloidBase()
: xCentre( 0 ), yCentre( 0 ), zCentre( 0 ),
  coefficient( 1 ), offset ( 0 )
{
}

template< typename Real >
bool tnlSDFParaboloidBase< Real >::setup( const tnlParameterContainer& parameters,
        								 const tnlString& prefix)
{
   this->xCentre = parameters.GetParameter< double >( "x-centre" );
   this->yCentre = parameters.GetParameter< double >( "y-centre" );
   this->zCentre = parameters.GetParameter< double >( "z-centre" );
   this->coefficient = parameters.GetParameter< double >( "coefficient" );
   this->offset = parameters.GetParameter< double >( "offset" );

   return true;
}

template< typename Real >
void tnlSDFParaboloidBase< Real >::setXCentre( const Real& xCentre )
{
   this->xCentre = xCentre;
}

template< typename Real >
Real tnlSDFParaboloidBase< Real >::getXCentre() const
{
   return this->xCentre;
}

template< typename Real >
void tnlSDFParaboloidBase< Real >::setYCentre( const Real& yCentre )
{
   this->yCentre = yCentre;
}

template< typename Real >
Real tnlSDFParaboloidBase< Real >::getYCentre() const
{
   return this->yCentre;
}
template< typename Real >
void tnlSDFParaboloidBase< Real >::setZCentre( const Real& zCentre )
{
   this->zCentre = zCentre;
}

template< typename Real >
Real tnlSDFParaboloidBase< Real >::getZCentre() const
{
   return this->zCentre;
}

template< typename Real >
void tnlSDFParaboloidBase< Real >::setCoefficient( const Real& amplitude )
{
   this->coefficient = coefficient;
}

template< typename Real >
Real tnlSDFParaboloidBase< Real >::getCoefficient() const
{
   return this->coefficient;
}

template< typename Real >
void tnlSDFParaboloidBase< Real >::setOffset( const Real& offset )
{
   this->offset = offset;
}

template< typename Real >
Real tnlSDFParaboloidBase< Real >::getOffset() const
{
   return this->offset;
}

template< typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder,
             typename Vertex >
Real tnlSDFParaboloid< 1, Real >::getValue( const Vertex& v,
        									const Real& time) const
{
   const Real& x = v.x();
   if( YDiffOrder != 0 || ZDiffOrder != 0 )
      return 0.0;
   if( XDiffOrder == 0 )
      return this->coefficient * ( ( x - this -> xCentre ) * ( x - this -> xCentre ) - this->offset*this->offset );
   if( XDiffOrder == 1 )
      return 2.0 * this->coefficient * ( x - this -> xCentre );
   return 0.0;
}


template< typename Real >
   template< int XDiffOrder,
             int YDiffOrder,
             int ZDiffOrder,
             typename Vertex >
Real tnlSDFParaboloid< 2, Real >::getValue( const Vertex& v,
        									const Real& time) const
{
   const Real& x = v.x();
   const Real& y = v.y();
   if( ZDiffOrder != 0 )
      return 0.0;
   if( XDiffOrder == 0 && YDiffOrder == 0 && ZDiffOrder == 0 )
   {
      return this->coefficient * ( ( x - this -> xCentre ) * ( x - this -> xCentre )
    		  	  	  	         + ( y - this -> yCentre ) * ( y - this -> yCentre ) - this->offset*this->offset );
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
             int ZDiffOrder,
             typename Vertex >
Real tnlSDFParaboloid< 3, Real >::getValue( const Vertex& v,
        									const Real& time) const
{
   const Real& x = v.x();
   const Real& y = v.y();
   const Real& z = v.z();
   if( XDiffOrder == 0 && YDiffOrder == 0 && ZDiffOrder == 0 )
   {
      return this->coefficient * ( ( x - this -> xCentre ) * ( x - this -> xCentre )
    		  	  	  	         + ( y - this -> yCentre ) * ( y - this -> yCentre )
    		  	  	  	         + ( z - this -> zCentre ) * ( z - this -> zCentre ) - this->offset*this->offset );
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

#endif /* TNLSDFPARABOLOID_IMPL_H_ */
