/***************************************************************************
                          tnlGridTopologies.h  -  description
                             -------------------
    begin                : Nov 13, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
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

#ifndef TNLGRIDTOPOLOGIES_H
#define	TNLGRIDTOPOLOGIES_H

template< typename Grid,
          int EntityDimenisons >
class tnlGridEntityTopology
{
   static_assert( false );
}

template< int Dimensions,
          typename Real,
          typename Device,
          typename Index >
class tnlGridCell< tnlGrid< Dimensions, Real, Device, Index > >
{
   public:
      static const int dimensions = Dimensions;
      
      static constexpr int getDimensions() { return dimensions; }
};

template< >


#endif	/* TNLGRIDTOPOLOGIES_H */

