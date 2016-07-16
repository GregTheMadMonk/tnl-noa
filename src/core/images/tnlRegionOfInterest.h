/***************************************************************************
                          tnlRegionOfInterest.h  -  description
                             -------------------
    begin                : Jul 22, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

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

