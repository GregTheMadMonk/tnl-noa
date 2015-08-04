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

#ifndef TNLREGIONOFINTEREST_IMPL_H
#define	TNLREGIONOFINTEREST_IMPL_H

#include "tnlImage.h"


template< typename Index >
tnlRegionOfInterest< Index >::
tnlRegionOfInterest()
: top( -1 ), bottom( -1 ), left( -1 ), right( -1 )
{   
}
      
template< typename Index >
bool
tnlRegionOfInterest< Index >::
setup( const tnlParameterContainer& parameters,
       const tnlImage< Index >* image )
{
   const int roiTop    = parameters.getParameter< int >( "roi-top" );
   const int roiBottom = parameters.getParameter< int >( "roi-bottom" );
   const int roiRight  = parameters.getParameter< int >( "roi-right" );
   const int roiLeft   = parameters.getParameter< int >( "roi-left" );
    
   if( roiBottom < roiTop )
   {
      cerr << "Error: roi-bottom (" << roiBottom << ") is smaller than roi-top (" << roiTop << ")." << endl;
      return false;
   }
   if( roiRight < roiLeft )
   {
      cerr << "Error: roi-right (" << roiRight << ") is smaller than roi-left (" << roiLeft << ")." << endl;
      return false;
   }

   if( roiLeft == -1 )
        this->left = 0;
   else
   {
      if( roiLeft >= image->getWidth() )
      {
         cerr << "ROI left column is larger than image width ( " << image->getWidth() << ")." << cerr;
         return false;
      }
      this->left = roiLeft;
   }
    
   if( roiRight == -1 )
      this->right = image->getWidth();
   else
   {
      if( roiRight >= image->getWidth() )
      {
         cerr << "ROI right column is larger than image width ( " << image->getWidth() << ")." << cerr;
         return false;
      }
      this->right = roiRight;
   }
    
   if( roiTop == -1 )
      this->top = 0;
   else
   {
      if( roiTop >= image->getHeight() )
      {
         cerr << "ROI top line is larger than image height ( " << image->getHeight() << ")." << cerr;
         return false;
      }
      this->top = roiTop;
   }
    
   if( roiBottom == -1 )
      this->bottom = image->getHeight();
   else
   {
      if( roiBottom >= image->getHeight() )
      {
         cerr << "ROI bottom line is larger than image height ( " << image->getHeight() << ")." << cerr;
         return false;
      }
      this->bottom = roiBottom;
   }
   return true;
}

template< typename Index >
bool
tnlRegionOfInterest< Index >::
check( const tnlImage< Index >* image ) const
{
   if( top >= image->getHeight() ||
       bottom >= image->getHeight() ||
       left >= image->getWidth() ||
       right >= image->getWidth() )
      return false;
   return true;
}

template< typename Index >
Index
tnlRegionOfInterest< Index >::
getTop() const
{
   return this->top;
}

template< typename Index >
Index
tnlRegionOfInterest< Index >::
getBottom() const
{
   return this->bottom;
}

template< typename Index >
Index
tnlRegionOfInterest< Index >::
getLeft() const
{
   return this->left;
}

template< typename Index >
Index
tnlRegionOfInterest< Index >::
getRight() const
{
   return this->right;
}

template< typename Index >
Index
tnlRegionOfInterest< Index >::
getWidth() const
{
   return this->right - this->left;
}

template< typename Index >
Index
tnlRegionOfInterest< Index >::
getHeight() const
{
   return this->bottom - this->top;
}

template< typename Index >
   template< typename Grid >
bool
tnlRegionOfInterest< Index >::
setGrid( Grid& grid,
         bool verbose )
{
    grid.setDimensions( this->getWidth(), this->getHeight() );
    typename Grid::VertexType origin, proportions;
    origin.x() = 0.0;
    origin.y() = 0.0;
    proportions.x() = 1.0;
    proportions.y() = ( double ) grid.getDimensions().x() / ( double ) grid.getDimensions().y();
    grid.setDomain( origin, proportions );
    if( verbose )
    {
        cout << "Setting grid to dimensions " << grid.getDimensions() << 
                " and proportions " << grid.getProportions() << endl;
    }
    return true;
}


template< typename Index >
bool
tnlRegionOfInterest< Index >::
isIn( const Index row, const Index column ) const
{
   if( row >= top && row < bottom &&
       column >= left && column < right )
      return true;
   return false;
}

#endif	/* TNLREGIONOFINTEREST_IMPL_H */

