/***************************************************************************
                          FiniteDifferences_impl.h  -  description
                             -------------------
    begin                : Nov 24, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Operators/FiniteDifferences.h>

namespace TNL {
namespace Operators {

template< typename Real, typename Device, typename Index >
   template< typename GridFunction,
             int XDifferenceOrder,
             int YDifferenceOrder,
             int ZDifferenceOrder,
             int XDifferenceDirection,
             int YDifferenceDirection,
             int ZDifferenceDirection >
Real FiniteDifferences< Meshes::Grid< 1, Real, Device, Index > >::getDifference( const GridType& grid,
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

   typename GridType::Cell cell( grid );
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
Real FiniteDifferences< Meshes::Grid< 1, Real, Device, Index > >::getDifference( const GridType& grid,
                                                                               const CellType& cell,
                                                                               const GridFunction& function )
{

   if( YDifferenceOrder > 0 || ZDifferenceOrder > 0 )
      return 0.0;
   const RealType hx = grid.getSpaceSteps().x();
   auto neighbourEntities = cell.getNeighbourEntities();
   IndexType cellIndex = grid.getEntityIndex( cell );
   if( XDifferenceOrder == 1 )
   {
      if( XDifferenceDirection == 0 )
         return ( function[ neighbourEntities.template getEntityIndex< 1 >() ] -
                  function[ neighbourEntities.template getEntityIndex< -1 >() ] ) / ( 2.0 * hx );
      else
         return ( function[ neighbourEntities.template getEntityIndex< XDifferenceDirection >() ] -
                  function[ cellIndex ] ) / ( XDifferenceDirection * hx );
   }
   if( XDifferenceOrder == 2 )
   {
      return ( function[ neighbourEntities.template getEntityIndex< 1 >() ] -
               2.0 * function[ cellIndex ] +
               function[ neighbourEntities.template getEntityIndex< -1 >() ] ) / (  hx * hx );
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
Real FiniteDifferences< Meshes::Grid< 2, Real, Device, Index > >::getDifference( const GridType& grid,
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
Real FiniteDifferences< Meshes::Grid< 2, Real, Device, Index > >::getDifference( const GridType& grid,
                                                                               const CellType& cell,
                                                                               const GridFunction& function )
{
   if( ZDifferenceOrder > 0 )
      return 0.0;
   auto neighbourEntities = cell.getNeighbourEntities();
   IndexType cellIndex = grid.getEntityIndex( cell );
   if( XDifferenceOrder == 1 )
   {
      const RealType hx = grid.getSpaceSteps().x();
      return ( function[ neighbourEntities.template getEntityIndex< XDifferenceDirection, 0 >( cellIndex ) ] -
               function[ cellIndex ] ) / ( XDifferenceDirection * hx );
   }
   if( XDifferenceOrder == 2 )
   {
      const RealType hx = grid.getSpaceSteps().x();
      return ( function[ neighbourEntities.template getEntityIndex< 1, 0 >( cellIndex ) ] -
               2.0 * function[ cellIndex ] +
               function[ neighbourEntities.template getEntityIndex< -1, 0 >( cellIndex ) ] ) / (  hx * hx );
   }
   if( YDifferenceOrder == 1 )
   {
      const RealType hy = grid.getSpaceSteps().y();
      return ( function[ neighbourEntities.template getEntityIndex< 0, YDifferenceDirection >( cellIndex ) ] -
               function[ cellIndex ] ) / ( YDifferenceDirection * hy );
   }
   if( YDifferenceOrder == 2 )
   {
      const RealType hy = grid.getSpaceSteps().y();
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
Real FiniteDifferences< Meshes::Grid< 3, Real, Device, Index > >::getDifference( const GridType& grid,
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
Real FiniteDifferences< Meshes::Grid< 3, Real, Device, Index > >::getDifference( const GridType& grid,
                                                                               const CellType& cell,
                                                                               const GridFunction& function )
{
   auto neighbourEntities = cell.getNeighbourEntities();
   IndexType cellIndex = grid.getEntityIndex( cell );

   if( XDifferenceOrder == 1 )
   {
      const RealType hx = grid.getSpaceSteps().x();
      return ( function[ neighbourEntities.template getEntityIndex< XDifferenceDirection, 0, 0 >( cellIndex ) ] -
               function[ cellIndex ] ) / ( XDifferenceDirection * hx );
   }
   if( XDifferenceOrder == 2 )
   {
      const RealType hx = grid.getSpaceSteps().x();
      return ( function[ neighbourEntities.template getEntityIndex< 1, 0, 0 >( cellIndex ) ] -
               2.0 * function[ cellIndex ] +
               function[ neighbourEntities.template getEntityIndex< -1, 0, 0 >( cellIndex ) ] ) / (  hx * hx );
   }
   if( YDifferenceOrder == 1 )
   {
      const RealType hy = grid.getSpaceSteps().y();
      return ( function[ neighbourEntities.template getEntityIndex< 0, YDifferenceDirection, 0 >( cellIndex ) ] -
               function[ cellIndex ] ) / ( YDifferenceDirection * hy );
   }
   if( YDifferenceOrder == 2 )
   {
      const RealType hy = grid.getSpaceSteps().y();
      return ( function[ neighbourEntities.template getEntityIndex< 0, 1, 0 >( cellIndex ) ] -
               2.0 * function[ cellIndex ] +
               function[ neighbourEntities.template getEntityIndex< 0, -1, 0 >( cellIndex ) ] ) / (  hy * hy );
   }
   if( ZDifferenceOrder == 1 )
   {
      const RealType hz = grid.getSpaceSteps().z();
      return ( function[ neighbourEntities.template getEntityIndex< 0, 0, ZDifferenceDirection >( cellIndex ) ] -
               function[ cellIndex ] ) / ( ZDifferenceDirection * hz );
   }
   if( ZDifferenceOrder == 2 )
   {
      const RealType hz = grid.getSpaceSteps().z();
      return ( function[ neighbourEntities.template getEntityIndex< 0, 0, 1 >( cellIndex ) ] -
               2.0 * function[ cellIndex ] +
               function[ neighbourEntities.template getEntityIndex< 0, 0, -1 >( cellIndex ) ] ) / (  hz * hz );
   }


}

} // namespace Operators
} // namespace TNL

#include <TNL/Operators/FiniteDifferences_impl.h>