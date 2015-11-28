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

#include <operators/tnlFiniteDifferences.h>

template< typename Real, typename Device, typename Index >
   template< typename GridFunction,
             int XDifferenceOrder,
             int YDifferenceOrder,
             int ZDifferenceOrder,
             int XDifferenceDirection,
             int YDifferenceDirection,
             int ZDifferenceDirection >
Real tnlFiniteDifferences< tnlGrid< 1, Real, Device, Index > >::getDifference( const GridType& grid,
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

   typename GridType::template GridEntity< GridType::Cells > cell( grid );
   for( cell.getCoordinates().x() = iBegin;
        cell.getCoordinates().x() < iEnd;
        cell.getCoordinates().x()++ )
   {
      outFunction[ grid.getEntityIndex( cell ) ] =
               getDifference< GridFunction,
                              XDifferenceOrder,
                              YDifferenceOrder,
                              ZDifferenceOrder,
                              XDifferenceDirection,
                              YDifferenceDirection,
                              ZDifferenceDirection >( grid, cell, inFunction );
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
Real tnlFiniteDifferences< tnlGrid< 1, Real, Device, Index > >::getDifference( const GridType& grid,
                                                                               const CellType& cell,
                                                                               const GridFunction& function )
{

   if( YDifferenceOrder > 0 || ZDifferenceOrder > 0 )
      return 0.0;
   const RealType hx = grid.getCellProportions().x();
   auto neighbourEntities = cell.getNeighbourEntities();
   IndexType cellIndex = grid.getEntityIndex( cell );
   if( XDifferenceOrder == 1 )
   {
      if( XDifferenceDirection == 0 )
         return ( function[ neighbourEntities.template getEntityIndex< 1 >( cellIndex ) ] -
                  function[ neighbourEntities.template getEntityIndex< -1 >( cellIndex ) ] ) / ( 2.0 * hx );
      else
         return ( function[ neighbourEntities.template getEntityIndex< XDifferenceDirection >( cellIndex ) ] -
                  function[ cellIndex ] ) / ( XDifferenceDirection * hx );
   }
   if( XDifferenceOrder == 2 )
   {
      return ( function[ neighbourEntities.template getEntityIndex< 1 >( cellIndex ) ] -
               2.0 * function[ cellIndex ] +
               function[ neighbourEntities.template getEntityIndex< -1 >( cellIndex ) ] ) / (  hx * hx );
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
Real tnlFiniteDifferences< tnlGrid< 2, Real, Device, Index > >::getDifference( const GridType& grid,
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
Real tnlFiniteDifferences< tnlGrid< 2, Real, Device, Index > >::getDifference( const GridType& grid,
                                                                               const CellType& cell,
                                                                               const GridFunction& function )
{
   if( ZDifferenceOrder > 0 )
      return 0.0;   
   auto neighbourEntities = cell.getNeighbourEntities();
   IndexType cellIndex = grid.getEntityIndex( cell );
   if( XDifferenceOrder == 1 )
   {
      const RealType hx = grid.getCellProportions().x();
      return ( function[ neighbourEntities.template getEntityIndex< XDifferenceDirection, 0 >( cellIndex ) ] -
               function[ cellIndex ] ) / ( XDifferenceDirection * hx );
   }
   if( XDifferenceOrder == 2 )
   {
      const RealType hx = grid.getCellProportions().x();
      return ( function[ neighbourEntities.template getEntityIndex< 1, 0 >( cellIndex ) ] -
               2.0 * function[ cellIndex ] +
               function[ neighbourEntities.template getEntityIndex< -1, 0 >( cellIndex ) ] ) / (  hx * hx );
   }
   if( YDifferenceOrder == 1 )
   {
      const RealType hy = grid.getCellProportions().y();
      return ( function[ neighbourEntities.template getEntityIndex< 0, YDifferenceDirection >( cellIndex ) ] -
               function[ cellIndex ] ) / ( YDifferenceDirection * hy );
   }
   if( YDifferenceOrder == 2 )
   {
      const RealType hy = grid.getCellProportions().y();
      return ( function[ neighbourEntities.template getEntityIndex< 0, 1 >( cellIndex ) ] -
               2.0 * function[ cellIndex ] +
               function[ neighbourEntities.template getEntityIndex< 0, -1 >( cellIndex ) ] ) / (  hy * hy );
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
Real tnlFiniteDifferences< tnlGrid< 3, Real, Device, Index > >::getDifference( const GridType& grid,
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
Real tnlFiniteDifferences< tnlGrid< 3, Real, Device, Index > >::getDifference( const GridType& grid,
                                                                               const CellType& cell,
                                                                               const GridFunction& function )
{
   auto neighbourEntities = cell.getNeighbourEntities();
   IndexType cellIndex = grid.getEntityIndex( cell );

   if( XDifferenceOrder == 1 )
   {
      const RealType hx = grid.getCellProportions().x();
      return ( function[ neighbourEntities.template getEntityIndex< XDifferenceDirection, 0, 0 >( cellIndex ) ] -
               function[ cellIndex ] ) / ( XDifferenceDirection * hx );
   }
   if( XDifferenceOrder == 2 )
   {
      const RealType hx = grid.getCellProportions().x();
      return ( function[ neighbourEntities.template getEntityIndex< 1, 0, 0 >( cellIndex ) ] -
               2.0 * function[ cellIndex ] +
               function[ neighbourEntities.template getEntityIndex< -1, 0, 0 >( cellIndex ) ] ) / (  hx * hx );
   }
   if( YDifferenceOrder == 1 )
   {      
      const RealType hy = grid.getCellProportions().y();
      return ( function[ neighbourEntities.template getEntityIndex< 0, YDifferenceDirection, 0 >( cellIndex ) ] -
               function[ cellIndex ] ) / ( YDifferenceDirection * hy );
   }
   if( YDifferenceOrder == 2 )
   {
      const RealType hy = grid.getCellProportions().y();
      return ( function[ neighbourEntities.template getEntityIndex< 0, 1, 0 >( cellIndex ) ] -
               2.0 * function[ cellIndex ] +
               function[ neighbourEntities.template getEntityIndex< 0, -1, 0 >( cellIndex ) ] ) / (  hy * hy );
   }
   if( ZDifferenceOrder == 1 )
   {
      const RealType hz = grid.getCellProportions().z();
      return ( function[ neighbourEntities.template getEntityIndex< 0, 0, ZDifferenceDirection >( cellIndex ) ] -
               function[ cellIndex ] ) / ( ZDifferenceDirection * hz );
   }
   if( ZDifferenceOrder == 2 )
   {
      const RealType hz = grid.getCellProportions().z();
      return ( function[ neighbourEntities.template getEntityIndex< 0, 0, 1 >( cellIndex ) ] -
               2.0 * function[ cellIndex ] +
               function[ neighbourEntities.template getEntityIndex< 0, 0, -1 >( cellIndex ) ] ) / (  hz * hz );
   }


}


#include <operators/tnlFiniteDifferences_impl.h>



#endif /* TNLFINITEDIFFERENCES_IMPL_H_ */
