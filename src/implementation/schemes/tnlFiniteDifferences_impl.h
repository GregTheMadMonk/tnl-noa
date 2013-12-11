/***************************************************************************
                          tnlFiniteDifferences_impl.h  -  description
                             -------------------
    begin                : Nov 24, 2013
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

#ifndef TNLFINITEDIFFERENCES_IMPL_H_
#define TNLFINITEDIFFERENCES_IMPL_H_

#include <schemes/tnlFiniteDifferences.h>

template< typename Real, typename Device, typename Index >
   template< typename GridFunction,
             int XDifferenceOrder,
             int YDifferenceOrder,
             int ZDifferenceOrder,
             int XDifferenceDirection,
             int YDifferenceDirection,
             int ZDifferenceDirection >
Real tnlFiniteDifferences< tnlGrid< 1, Real, Device, Index, tnlIdenticalGridGeometry > >::getDifference( const GridType& grid,
                                                                                                         const GridFunction& inFunction,
                                                                                                         GridFunction& outFunction )
{
   IndexType iBegin, iEnd;
   if( XDifferenceDirection == 0 || XDifferenceDirection == 1 )
      iBegin = 0;
   else
      iBegin = 1;
   if( XDifferenceDirection == 1 )
      iEnd = grid.getDimensions().x() - 1;
   else
      iEnd = grid.getDimensions().x();

   CoordinatesType c;
   for( c.x() = iBegin; c.x() < iEnd; c.x()++ )
   {
      outFunction[ grid.getElementIndex( c.x() ) ] =
               getDifference< GridFunction,
                              XDifferenceOrder,
                              YDifferenceOrder,
                              ZDifferenceOrder,
                              XDifferenceDirection,
                              YDifferenceDirection,
                              ZDifferenceDirection >( grid, c, inFunction );
   }
}

template< typename Real, typename Device, typename Index >
   template< typename GridFunction,
             int XDifferenceOrder,
             int YDifferenceOrder,
             int ZDifferenceOrder,
             int XDifferenceDirection,
             int YDifferenceDirection,
             int ZDifferenceDirection >
Real tnlFiniteDifferences< tnlGrid< 1, Real, Device, Index, tnlIdenticalGridGeometry > >::getDifference( const GridType& grid,
                                                                                                         const CoordinatesType& c,
                                                                                                         const GridFunction& function )
{

   if( YDifferenceOrder > 0 || ZDifferenceOrder > 0 )
      return 0.0;
   const RealType hx = grid.getParametricStep().x();
   if( XDifferenceOrder == 1 )
   {
      if( XDifferenceDirection == 0 )
         return ( function[ grid.getElementIndex( c.x() + 1 ) ] -
                  function[ grid.getElementIndex( c.x() - 1 ) ] ) / ( 2.0 * hx );
      else
         return ( function[ grid.getElementIndex( c.x() + XDifferenceDirection ) ] -
                  function[ grid.getElementIndex( c.x() ) ] ) / ( XDifferenceDirection * hx );
   }
   if( XDifferenceOrder == 2 )
   {
      return ( function[ grid.getElementIndex( c.x() + 1 ) ] -
               2.0 * function[ grid.getElementIndex( c.x() ) ] +
               function[ grid.getElementIndex( c.x() - 1 ) ] ) / (  hx * hx );
   }
}

/****
 *  2D Grid
 */

template< typename Real, typename Device, typename Index >
   template< typename GridFunction,
             int XDifferenceOrder,
             int YDifferenceOrder,
             int ZDifferenceOrder,
             int XDifferenceDirection,
             int YDifferenceDirection,
             int ZDifferenceDirection >
Real tnlFiniteDifferences< tnlGrid< 2, Real, Device, Index, tnlIdenticalGridGeometry > >::getDifference( const GridType& grid,
                                                                                                         const GridFunction& inFunction,
                                                                                                         GridFunction& outFunction )
{

}

template< typename Real, typename Device, typename Index >
   template< typename GridFunction,
             int XDifferenceOrder,
             int YDifferenceOrder,
             int ZDifferenceOrder,
             int XDifferenceDirection,
             int YDifferenceDirection,
             int ZDifferenceDirection >
Real tnlFiniteDifferences< tnlGrid< 2, Real, Device, Index, tnlIdenticalGridGeometry > >::getDifference( const GridType& grid,
                                                                                                         const CoordinatesType& c,
                                                                                                         const GridFunction& function )
{
   if( YDifferenceOrder > 0 || ZDifferenceOrder > 0 )
      return 0.0;
   const RealType hx = grid.getParametricSpaceStep();
   if( XDifferenceOrder == 1 )
   {
      return ( function[ grid.getElementIndex( c.x() + XDifferenceDirection ) ] -
               function[ grid.getElementIndex( c.x() ) ] ) / ( XDifferenceDirection * hx );
   }
   if( XDifferenceOrder == 2 )
   {
      return ( function[ grid.getElementIndex( c.x() + 1 ) ] -
               2.0 * function[ grid.getElementIndex( c.x() ) ] +
               function[ grid.getElementIndex( c.x() - 1 ) ] ) / (  hx * hx );
   }

}

/****
 *  3D Grid
 */

template< typename Real, typename Device, typename Index >
   template< typename GridFunction,
             int XDifferenceOrder,
             int YDifferenceOrder,
             int ZDifferenceOrder,
             int XDifferenceDirection,
             int YDifferenceDirection,
             int ZDifferenceDirection >
Real tnlFiniteDifferences< tnlGrid< 3, Real, Device, Index, tnlIdenticalGridGeometry > >::getDifference( const GridType& grid,
                                                                                                         const GridFunction& inFunction,
                                                                                                         GridFunction& outFunction )
{

}

template< typename Real, typename Device, typename Index >
   template< typename GridFunction,
             int XDifferenceOrder,
             int YDifferenceOrder,
             int ZDifferenceOrder,
             int XDifferenceDirection,
             int YDifferenceDirection,
             int ZDifferenceDirection >
Real tnlFiniteDifferences< tnlGrid< 3, Real, Device, Index, tnlIdenticalGridGeometry > >::getDifference( const GridType& grid,
                                                                                                         const CoordinatesType& c,
                                                                                                         const GridFunction& function )
{
   if( YDifferenceOrder > 0 || ZDifferenceOrder > 0 )
      return 0.0;
   const RealType hx = grid.getParametricSpaceStep();
   if( XDifferenceOrder == 1 )
   {
      return ( function[ grid.getElementIndex( c.x() + XDifferenceDirection ) ] -
               function[ grid.getElementIndex( c.x() ) ] ) / ( XDifferenceDirection * hx );
   }
   if( XDifferenceOrder == 2 )
   {
      return ( function[ grid.getElementIndex( c.x() + 1 ) ] -
               2.0 * function[ grid.getElementIndex( c.x() ) ] +
               function[ grid.getElementIndex( c.x() - 1 ) ] ) / (  hx * hx );
   }

}


#include <implementation/schemes/tnlFiniteDifferences_impl.h>



#endif /* TNLFINITEDIFFERENCES_IMPL_H_ */
