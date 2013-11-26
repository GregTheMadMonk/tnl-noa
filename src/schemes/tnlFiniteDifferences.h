/***************************************************************************
                          tnlFiniteDifferences.h  -  description
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

#ifndef TNLFINITEDIFFERENCES_H_
#define TNLFINITEDIFFERENCES_H_

#include <mesh/tnlGrid.h>
#include <mesh/tnlIdenticalGridGeometry.h>

template< typename Grid >
class tnlFiniteDifferences
{
};

template< typename Real = double, typename Device = tnlHost, typename Index = int >
class tnlFiniteDifferences< tnlGrid< 1, Real, Device, Index, tnlIdenticalGridGeometry > >
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef tnlTuple< 1, Index > CoordinatesType;
   typedef tnlGrid< 1, Real, Device, Index, tnlIdenticalGridGeometry > GridType;

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
                                  const CoordinatesType& c,
                                  const GridFunction& function );

};

#include <implementation/schemes/tnlFiniteDifferences_impl.h>



#endif /* TNLFINITEDIFFERENCES_H_ */
