/***************************************************************************
                          tnlRegionOfInterest.h  -  description
                             -------------------
    begin                : Jul 22, 2015
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

#ifndef TNLREGIONOFINTEREST_H
#define	TNLREGIONOFINTEREST_H

#include <config/tnlParameterContainer.h>
#include <mesh/tnlGrid.h>
#include <core/images/tnlImage.h>


template< typename Index = int >
class tnlRegionOfInterest
{
   public:
      
      tnlRegionOfInterest();
      
      bool setup( const tnlParameterContainer& parameters,
                  const tnlImage< Index >* image );
      
      bool check( const tnlImage< Index >* image ) const;
      
      Index getTop() const;
      
      Index getBottom() const;
      
      Index getLeft() const;
      
      Index getRight() const;
      
      Index getWidth() const;
      
      Index getHeight() const;
      
      template< typename Grid >
         bool setGrid( Grid& grid,
                       bool verbose = false );
      
      bool isIn( const Index row, const Index column ) const;
      
   protected:
      
      Index top, bottom, left, right;
};

#include <core/images/tnlRegionOfInterest_impl.h>

#endif	/* TNLREGIONOFINTEREST_H */

