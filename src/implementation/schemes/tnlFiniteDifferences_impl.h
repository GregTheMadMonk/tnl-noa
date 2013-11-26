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
   template< typename GridFunction >
      Real tnlFiniteDifferences< tnlGrid< 1, Real, Device, Index, tnlIdenticalGridGeometry > >::
         getFirstForwardDifferenceX( const GridType& grid,
                                     const CoordinatesType& c,
                                     const GridFunction& function )
{
   tnlAssert( c.x() < grid.getDimensions().x() - 1,
             cerr << "c.x() = " << c.x() << " grid.getDimensions().x() - 1 = " << grid.getDimensions().x() - 1 );
   Real hx = grid.getParametricStep().x();
   Index i1 = grid.getElementIndex( c.x() );
   Index i2 = grid.getElementIndex( c.x() + 1 );
   return 1.0 / hx * ( function[ i2 ] - function[ i1 ] );
}

template< typename Real, typename Device, typename Index >
   template< typename GridFunction >
      void tnlFiniteDifferences< tnlGrid< 1, Real, Device, Index, tnlIdenticalGridGeometry > >::
         getFirstForwardDifferenceX( const GridType& grid,
                                     const GridFunction& inFunction,
                                     GridFunction& outFunction )
{
   for( Index i = 0; i < grid.getDimensions().x() - 1; i++ )
   {
      CoordinatesType c;
      c.x() = i;
      Index k = grid.getElementIndex( c );
      outFunction[ k ] = this->getFirstForwardDifferenceX( grid, c, inFunciton );
   }
}

template< typename Real, typename Device, typename Index >
   template< typename GridFunction >
      Real tnlFiniteDifferences< tnlGrid< 1, Real, Device, Index, tnlIdenticalGridGeometry > >::
         getFirstBackwardDifferenceX( const GridType& grid,
                                      const CoordinatesType& c,
                                      GridFunction& function )
{

}

template< typename Real, typename Device, typename Index >
   template< typename GridFunction >
      Real tnlFiniteDifferences< tnlGrid< 1, Real, Device, Index, tnlIdenticalGridGeometry > >::
         getFirstBackwardDifferenceX( const GridType& grid,
                                      GridFunction& function )
{

}

template< typename Real, typename Device, typename Index >
   template< typename GridFunction >
      Real tnlFiniteDifferences< tnlGrid< 1, Real, Device, Index, tnlIdenticalGridGeometry > >::
         getFirstCentralDifferenceX( const GridType& grid,
                                     const CoordinatesType& c,
                                     GridFunction& function )
{

}

template< typename Real, typename Device, typename Index >
   template< typename GridFunction >
      Real tnlFiniteDifferences< tnlGrid< 1, Real, Device, Index, tnlIdenticalGridGeometry > >::
         getFirstCentralDifferenceX( const GridType& grid,
                                     GridFunction& function )
{
}

template< typename Real, typename Device, typename Index >
   template< typename GridFunction >
      Real tnlFiniteDifferences< tnlGrid< 1, Real, Device, Index, tnlIdenticalGridGeometry > >::
         getSecondCentralDifferenceX( const GridType& grid,
                                      const CoordinatesType& c,
                                      GridFunction& function )
{

}

template< typename Real, typename Device, typename Index >
   template< typename GridFunction >
      Real tnlFiniteDifferences< tnlGrid< 1, Real, Device, Index, tnlIdenticalGridGeometry > >::
         getSecondCentralDifferenceX( const GridType& grid,
                                      GridFunction& function )
{
}

#include <implementation/schemes/tnlFiniteDifferences_impl.h>



#endif /* TNLFINITEDIFFERENCES_IMPL_H_ */
