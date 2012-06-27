/***************************************************************************
                          tnlDistributedGrid.h  -  description
                             -------------------
    begin                : Feb 26, 2011
    copyright            : (C) 2011 by Tomas Oberhuber
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

#ifndef TNLDISTRIBUTEDGRID_H_
#define TNLDISTRIBUTEDGRID_H_

#include <core/tnlObject.h>
#include <core/tnlCommunicator.h>

template< int Dimensions,
          typename GridType,
          tnlDevice Device = tnlHost,
          typename Real = double,
          typename Index = int >
class tnlDistributedGrid : public tnlObject
{
   //! We do not allow constructor without parameters.
   tnlDistributedGrid();

   //! We do not allow copy constructor without object name.
   tnlDistributedGrid( const tnlDistributedGrid< Dimensions, Real, Device, Index >& a );

   public:

   tnlDistributedGrid( const tnlString& name );

   bool init( tnlCommunicator* communicator,
              const GridType& grid,
              const tnlTuple< Dimensions, Index >& subdomainOverlaps );

   tnlCommunicator< Device >* getCommunicator() const;

   const tnlTuple< Dimensions, Real >& getDomainLowerCorner() const;

   const tnlTuple< Dimensions, Real >& getDomainUpperCorner() const;

   const tnlTuple< Dimensions, Index >& getDimensions() const;

   const tnlTuple< Dimensions, int >& getGridDimensions() const;

   const tnlTuple< Dimensions, int >& getLowerNeighbors() const;

   const tnlTuple< Dimensions, Index >& getLowerSubdomainsOverlaps() const;

   const tnlTuple< Dimensions, int >& getNodeCoordinates() const;

   const tnlTuple< Dimensions, Index >& getSubdomainDimensions() const;

   const tnlTuple< Dimensions, Index >& getUpperSubdomainsOverlaps() const;

   const tnlTuple< Dimensions, int >& getUppperNeighbors() const;

   protected:

   //! Pointer to the communicator used by this distributed grid.
   tnlCommunicator< Device >* communicator;

   //! In 2D this is the left bottom corner of the global domain.
   /*!***
    * This is naturally generalized to more dimensions.
    */
   tnlTuple< Dimensions, Real > domainLowerCorner;

   //! In 2D this is the right top corner of the global domain.
   /*!***
    * This is naturally generalized to more dimensions.
    */
   tnlTuple< Dimensions, Real > domainUpperCorner;

   //! Dimensions of the global domain.
   tnlTuple< Dimensions, Index > globalDimensions;

   //! Dimensions of the local subdomain.
   tnlTuple< Dimensions, Index > subdomainDimensions;

   //! Number of the distributed grid nodes along each dimension.
   tnlTuple< Dimensions, int > gridDimensions;

   //! Coordinates of this node of the distributed grid.
   tnlTuple< Dimensions, int > nodeCoordinates;

   //! Here are device IDs taken from the tnlCommunicator.
   /*!***
    * In 2D, this is the device ID of the neighbor on the
    * right and above.
    */
   tnlTuple< Dimensions, int > uppperNeighbors;

   //! Here are device IDs taken from the tnlCommunicator.
   /*!***
    * In 2D, this is the device ID of the neighbor on the
    * left and below.
    */
   tnlTuple< Dimensions, int > lowerNeighbors;

   //! Here are widths of overlaps at subdomain boundaries with neighbors.
   /*!***
    * These overlaps are necessary for exchange of data
    * between neighboring nodes. In 2D, here are overlaps
    * with the neighbors on the right and above.
    */
   tnlTuple< Dimensions, Index > upperSubdomainsOverlaps;

   //! Here are widths of overlaps at subdomain boundaries with neighbors.
   /*!***
    * These overlaps are necessary for exchange of data
    * between neighboring nodes. In 2D, here are overlaps
    * with the neighbors on the left and below.
    */
   tnlTuple< Dimensions, Index > lowerSubdomainsOverlaps;

};

template< int Dimensions, typename GridType, tnlDevice Device, typename Real, typename Index >
tnlDistributedGrid< Dimensions, GridType, Device, Real, Index > :: tnlDistributedGrid( const tnlString& name )
 : tnlObject( name )
{

}

template< int Dimensions, typename GridType, tnlDevice Device, typename Real, typename Index >
bool tnlDistributedGrid< Dimensions, GridType, Device, Real, Index > :: init( tnlCommunicator* communicator,
                                                                              const GridType& grid,
                                                                              const tnlTuple< Dimensions, int >& gridDimensions,
                                                                              const tnlTuple< Dimensions, Index >& subdomainOverlaps )
{

}

template< int Dimensions, typename GridType, tnlDevice Device, typename Real, typename Index >
tnlCommunicator* tnlDistributedGrid< Dimensions, GridType, Device, Real, Index > :: getCommunicator() const
{
    return communicator;
}

template< int Dimensions, typename GridType, tnlDevice Device, typename Real, typename Index >
const tnlTuple< Dimensions, Real >& tnlDistributedGrid< Dimensions, GridType, Device, Real, Index > :: getDomainLowerCorner() const
{
    return domainLowerCorner;
}

template< int Dimensions, typename GridType, tnlDevice Device, typename Real, typename Index >
const tnlTuple< Dimensions, Real >& tnlDistributedGrid< Dimensions, GridType, Device, Real, Index > :: getDomainUpperCorner() const
{
    return domainUpperCorner;
}

template< int Dimensions, typename GridType, tnlDevice Device, typename Real, typename Index >
const tnlTuple< Dimensions, Index >& tnlDistributedGrid< Dimensions, GridType, Device, Real, Index > :: getDimensions() const
{
    return globalDimensions;
}

template< int Dimensions, typename GridType, tnlDevice Device, typename Real, typename Index >
const tnlTuple< Dimensions, int >& tnlDistributedGrid< Dimensions, GridType, Device, Real, Index > :: getGridDimensions() const
{
    return gridDimensions;
}

template< int Dimensions, typename GridType, tnlDevice Device, typename Real, typename Index >
const tnlTuple< Dimensions, int >& tnlDistributedGrid< Dimensions, GridType, Device, Real, Index > :: getLowerNeighbors() const
{
    return lowerNeighbors;
}

template< int Dimensions, typename GridType, tnlDevice Device, typename Real, typename Index >
const tnlTuple< Dimensions, Index >& tnlDistributedGrid< Dimensions, GridType, Device, Real, Index > :: getLowerSubdomainsOverlaps() const
{
    return lowerSubdomainsOverlaps;
}

template< int Dimensions, typename GridType, tnlDevice Device, typename Real, typename Index >
const tnlTuple< Dimensions, int >& tnlDistributedGrid< Dimensions, GridType, Device, Real, Index > :: getNodeCoordinates() const
{
    return nodeCoordinates;
}

template< int Dimensions, typename GridType, tnlDevice Device, typename Real, typename Index >
const tnlTuple< Dimensions, Index >& tnlDistributedGrid< Dimensions, GridType, Device, Real, Index > :: getSubdomainDimensions() const
{
    return subdomainDimensions;
}

template< int Dimensions, typename GridType, tnlDevice Device, typename Real, typename Index >
const tnlTuple< Dimensions, Index >& tnlDistributedGrid< Dimensions, GridType, Device, Real, Index > :: getUpperSubdomainsOverlaps() const
{
    return upperSubdomainsOverlaps;
}

template< int Dimensions, typename GridType, tnlDevice Device, typename Real, typename Index >
const tnlTuple< Dimensions, int >& tnlDistributedGrid< Dimensions, GridType, Device, Real, Index > :: getUppperNeighbors() const
{
    return uppperNeighbors;
}

#endif /* TNLDISTRIBUTEDGRID_H_ */
