/***************************************************************************
                          tnlDistributedGrid.h  -  description
                             -------------------
    begin                : Feb 26, 2011
    copyright            : (C) 2011 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLDISTRIBUTEDGRID_H_
#define TNLDISTRIBUTEDGRID_H_

#include <TNL/Object.h>
#include <TNL/core/tnlCommunicator.h>

template< int Dimensions,
          typename GridType,
          typename Device = tnlHost,
          typename Real = double,
          typename Index = int >
class tnlDistributedGrid : public Object
{
   //! We do not allow constructor without parameters.
   tnlDistributedGrid();

   //! We do not allow copy constructor without object name.
   tnlDistributedGrid( const tnlDistributedGrid< Dimensions, Real, Device, Index >& a );

   public:

   tnlDistributedGrid( const String& name );

   bool init( tnlCommunicator* communicator,
              const GridType& grid,
              const StaticVector< Dimensions, Index >& subdomainOverlaps );

   tnlCommunicator< Device >* getCommunicator() const;

   const StaticVector< Dimensions, Real >& getDomainLowerCorner() const;

   const StaticVector< Dimensions, Real >& getDomainUpperCorner() const;

   const StaticVector< Dimensions, Index >& getDimensions() const;

   const StaticVector< Dimensions, int >& getGridDimensions() const;

   const StaticVector< Dimensions, int >& getLowerNeighbors() const;

   const StaticVector< Dimensions, Index >& getLowerSubdomainsOverlaps() const;

   const StaticVector< Dimensions, int >& getNodeCoordinates() const;

   const StaticVector< Dimensions, Index >& getSubdomainDimensions() const;

   const StaticVector< Dimensions, Index >& getUpperSubdomainsOverlaps() const;

   const StaticVector< Dimensions, int >& getUppperNeighbors() const;

   protected:

   //! Pointer to the communicator used by this distributed grid.
   tnlCommunicator< Device >* communicator;

   //! In 2D this is the left bottom corner of the global domain.
   /*!***
    * This is naturally generalized to more dimensions.
    */
   StaticVector< Dimensions, Real > domainLowerCorner;

   //! In 2D this is the right top corner of the global domain.
   /*!***
    * This is naturally generalized to more dimensions.
    */
   StaticVector< Dimensions, Real > domainUpperCorner;

   //! Dimensions of the global domain.
   StaticVector< Dimensions, Index > globalDimensions;

   //! Dimensions of the local subdomain.
   StaticVector< Dimensions, Index > subdomainDimensions;

   //! Number of the distributed grid nodes along each dimension.
   StaticVector< Dimensions, int > gridDimensions;

   //! Coordinates of this node of the distributed grid.
   StaticVector< Dimensions, int > nodeCoordinates;

   //! Here are device IDs taken from the tnlCommunicator.
   /*!***
    * In 2D, this is the device ID of the neighbor on the
    * right and above.
    */
   StaticVector< Dimensions, int > uppperNeighbors;

   //! Here are device IDs taken from the tnlCommunicator.
   /*!***
    * In 2D, this is the device ID of the neighbor on the
    * left and below.
    */
   StaticVector< Dimensions, int > lowerNeighbors;

   //! Here are widths of overlaps at subdomain boundaries with neighbors.
   /*!***
    * These overlaps are necessary for exchange of data
    * between neighboring nodes. In 2D, here are overlaps
    * with the neighbors on the right and above.
    */
   StaticVector< Dimensions, Index > upperSubdomainsOverlaps;

   //! Here are widths of overlaps at subdomain boundaries with neighbors.
   /*!***
    * These overlaps are necessary for exchange of data
    * between neighboring nodes. In 2D, here are overlaps
    * with the neighbors on the left and below.
    */
   StaticVector< Dimensions, Index > lowerSubdomainsOverlaps;

};

template< int Dimensions, typename GridType, typename Device, typename Real, typename Index >
tnlDistributedGrid< Dimensions, GridType, Device, Real, Index > :: tnlDistributedGrid( const String& name )
 : Object( name )
{

}

template< int Dimensions, typename GridType, typename Device, typename Real, typename Index >
bool tnlDistributedGrid< Dimensions, GridType, Device, Real, Index > :: init( tnlCommunicator* communicator,
                                                                              const GridType& grid,
                                                                              const StaticVector< Dimensions, int >& gridDimensions,
                                                                              const StaticVector< Dimensions, Index >& subdomainOverlaps )
{

}

template< int Dimensions, typename GridType, typename Device, typename Real, typename Index >
tnlCommunicator* tnlDistributedGrid< Dimensions, GridType, Device, Real, Index > :: getCommunicator() const
{
    return communicator;
}

template< int Dimensions, typename GridType, typename Device, typename Real, typename Index >
const StaticVector< Dimensions, Real >& tnlDistributedGrid< Dimensions, GridType, Device, Real, Index > :: getDomainLowerCorner() const
{
    return domainLowerCorner;
}

template< int Dimensions, typename GridType, typename Device, typename Real, typename Index >
const StaticVector< Dimensions, Real >& tnlDistributedGrid< Dimensions, GridType, Device, Real, Index > :: getDomainUpperCorner() const
{
    return domainUpperCorner;
}

template< int Dimensions, typename GridType, typename Device, typename Real, typename Index >
const StaticVector< Dimensions, Index >& tnlDistributedGrid< Dimensions, GridType, Device, Real, Index > :: getDimensions() const
{
    return globalDimensions;
}

template< int Dimensions, typename GridType, typename Device, typename Real, typename Index >
const StaticVector< Dimensions, int >& tnlDistributedGrid< Dimensions, GridType, Device, Real, Index > :: getGridDimensions() const
{
    return gridDimensions;
}

template< int Dimensions, typename GridType, typename Device, typename Real, typename Index >
const StaticVector< Dimensions, int >& tnlDistributedGrid< Dimensions, GridType, Device, Real, Index > :: getLowerNeighbors() const
{
    return lowerNeighbors;
}

template< int Dimensions, typename GridType, typename Device, typename Real, typename Index >
const StaticVector< Dimensions, Index >& tnlDistributedGrid< Dimensions, GridType, Device, Real, Index > :: getLowerSubdomainsOverlaps() const
{
    return lowerSubdomainsOverlaps;
}

template< int Dimensions, typename GridType, typename Device, typename Real, typename Index >
const StaticVector< Dimensions, int >& tnlDistributedGrid< Dimensions, GridType, Device, Real, Index > :: getNodeCoordinates() const
{
    return nodeCoordinates;
}

template< int Dimensions, typename GridType, typename Device, typename Real, typename Index >
const StaticVector< Dimensions, Index >& tnlDistributedGrid< Dimensions, GridType, Device, Real, Index > :: getSubdomainDimensions() const
{
    return subdomainDimensions;
}

template< int Dimensions, typename GridType, typename Device, typename Real, typename Index >
const StaticVector< Dimensions, Index >& tnlDistributedGrid< Dimensions, GridType, Device, Real, Index > :: getUpperSubdomainsOverlaps() const
{
    return upperSubdomainsOverlaps;
}

template< int Dimensions, typename GridType, typename Device, typename Real, typename Index >
const StaticVector< Dimensions, int >& tnlDistributedGrid< Dimensions, GridType, Device, Real, Index > :: getUppperNeighbors() const
{
    return uppperNeighbors;
}

#endif /* TNLDISTRIBUTEDGRID_H_ */
