/***************************************************************************
                          FiniteDifferences.h  -  description
                             -------------------
    begin                : Nov 24, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Meshes/Grid.h>

namespace TNL {
namespace Operators {

template< typename Grid >
class FiniteDifferences
{
};

template< typename Real, typename Device, typename Index >
class FiniteDifferences< Meshes::Grid< 1, Real, Device, Index > >
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef Meshes::Grid< 1, Real, Device, Index > GridType;
   //typedef typename GridType::CoordinatesType CoordinatesType;
   typedef typename GridType::Cell CellType;


   template< typename GridFunction,
             int XDifferenceOrder,
             int YDifferenceOrder,
             int ZDifferenceOrder,
             int XDifferenceDirection = 0,
             int YDifferenceDirection = 0,
             int ZDifferenceDirection = 0 >
   static RealType getDifference( const GridType& grid,
                                  const GridFunction& inFunction,
                                  GridFunction& outFunction );

   template< typename GridFunction,
             int XDifferenceOrder,
             int YDifferenceOrder,
             int ZDifferenceOrder,
             int XDifferenceDirection = 0,
             int YDifferenceDirection = 0,
             int ZDifferenceDirection = 0 >
   static RealType getDifference( const GridType& grid,
                                  const CellType& cell,
                                  const GridFunction& function );
};

template< typename Real, typename Device, typename Index >
class FiniteDifferences< Meshes::Grid< 2, Real, Device, Index > >
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef Meshes::Grid< 2, Real, Device, Index > GridType;
   //typedef typename GridType::CoordinatesType CoordinatesType;
   typedef typename GridType::Cell CellType;


   template< typename GridFunction,
             int XDifferenceOrder,
             int YDifferenceOrder,
             int ZDifferenceOrder,
             int XDifferenceDirection = 0,
             int YDifferenceDirection = 0,
             int ZDifferenceDirection = 0 >
   static RealType getDifference( const GridType& grid,
                                  const GridFunction& inFunction,
                                  GridFunction& outFunction );

   template< typename GridFunction,
             int XDifferenceOrder,
             int YDifferenceOrder,
             int ZDifferenceOrder,
             int XDifferenceDirection = 0,
             int YDifferenceDirection = 0,
             int ZDifferenceDirection = 0 >
   static RealType getDifference( const GridType& grid,
                                  const CellType& cell,
                                  const GridFunction& function );
};

template< typename Real, typename Device, typename Index >
class FiniteDifferences< Meshes::Grid< 3, Real, Device, Index > >
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef Meshes::Grid< 3, Real, Device, Index > GridType;
   //typedef typename GridType::CoordinatesType CoordinatesType;
   typedef typename GridType::Cell CellType;

   template< typename GridFunction,
             int XDifferenceOrder,
             int YDifferenceOrder,
             int ZDifferenceOrder,
             int XDifferenceDirection = 0,
             int YDifferenceDirection = 0,
             int ZDifferenceDirection = 0 >
   static RealType getDifference( const GridType& grid,
                                  const GridFunction& inFunction,
                                  GridFunction& outFunction );

   template< typename GridFunction,
             int XDifferenceOrder,
             int YDifferenceOrder,
             int ZDifferenceOrder,
             int XDifferenceDirection = 0,
             int YDifferenceDirection = 0,
             int ZDifferenceDirection = 0 >
   static RealType getDifference( const GridType& grid,
                                  const CellType& cell,
                                  const GridFunction& function );
};

} // namespace Operators
} // namespace TNL

#include <TNL/Operators/FiniteDifferences_impl.h>
