/***************************************************************************
                          FiniteDifferences.h  -  description
                             -------------------
    begin                : Nov 24, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/mesh/tnlGrid.h>

namespace TNL {
namespace Operators {

template< typename Grid >
class FiniteDifferences
{
};

template< typename Real, typename Device, typename Index >
class FiniteDifferences< tnlGrid< 1, Real, Device, Index > >
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef tnlGrid< 1, Real, Device, Index > GridType;
   //typedef typename GridType::CoordinatesType CoordinatesType;
   typedef typename GridType::Cell CellType;


#ifdef HAVE_NOT_CXX11
   template< typename GridFunction,
             int XDifferenceOrder,
             int YDifferenceOrder,
             int ZDifferenceOrder,
             int XDifferenceDirection,
             int YDifferenceDirection,
             int ZDifferenceDirection >
   static RealType getDifference( const GridType& grid,
                                  const GridFunction& inFunction,
                                  GridFunction& outFunction );
#else
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
#endif

#ifdef HAVE_NOT_CXX11
   template< typename GridFunction,
             int XDifferenceOrder,
             int YDifferenceOrder,
             int ZDifferenceOrder,
             int XDifferenceDirection,
             int YDifferenceDirection,
             int ZDifferenceDirection >
   static RealType getDifference( const GridType& grid,
                                  const CellType& cell,
                                  const GridFunction& function );
#else
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
#endif

};

template< typename Real, typename Device, typename Index >
class FiniteDifferences< tnlGrid< 2, Real, Device, Index > >
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef tnlGrid< 2, Real, Device, Index > GridType;
   //typedef typename GridType::CoordinatesType CoordinatesType;
   typedef typename GridType::Cell CellType;


#ifdef HAVE_NOT_CXX11
   template< typename GridFunction,
             int XDifferenceOrder,
             int YDifferenceOrder,
             int ZDifferenceOrder,
             int XDifferenceDirection,
             int YDifferenceDirection,
             int ZDifferenceDirection >
   static RealType getDifference( const GridType& grid,
                                  const GridFunction& inFunction,
                                  GridFunction& outFunction );
#else
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
#endif

#ifdef HAVE_NOT_CXX11
   template< typename GridFunction,
             int XDifferenceOrder,
             int YDifferenceOrder,
             int ZDifferenceOrder,
             int XDifferenceDirection,
             int YDifferenceDirection,
             int ZDifferenceDirection >
   static RealType getDifference( const GridType& grid,
                                  const CellType& cell,
                                  const GridFunction& function );
#else
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
#endif

};

template< typename Real, typename Device, typename Index >
class FiniteDifferences< tnlGrid< 3, Real, Device, Index > >
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef tnlGrid< 3, Real, Device, Index > GridType;
   //typedef typename GridType::CoordinatesType CoordinatesType;
   typedef typename GridType::Cell CellType;

#ifdef HAVE_NOT_CXX11
   template< typename GridFunction,
             int XDifferenceOrder,
             int YDifferenceOrder,
             int ZDifferenceOrder,
             int XDifferenceDirection,
             int YDifferenceDirection,
             int ZDifferenceDirection >
   static RealType getDifference( const GridType& grid,
                                  const GridFunction& inFunction,
                                  GridFunction& outFunction );
#else
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
#endif

#ifdef HAVE_NOT_CXX11
   template< typename GridFunction,
             int XDifferenceOrder,
             int YDifferenceOrder,
             int ZDifferenceOrder,
             int XDifferenceDirection,
             int YDifferenceDirection,
             int ZDifferenceDirection >
   static RealType getDifference( const GridType& grid,
                                  const CellType& cell,
                                  const GridFunction& function );
#else
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
#endif

};

} // namespace Operators
} // namespace TNL

#include <TNL/Operators/FiniteDifferences_impl.h>