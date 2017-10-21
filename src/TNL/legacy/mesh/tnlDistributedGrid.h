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
#include <TNL/tnlCommunicator.h>

template< int Dimension,
          typename GridType,
          typename Device = Devices::Host,
          typename Real = double,
          typename Index = int >
class tnlDistributedGrid : public Object
{
   //! We do not allow constructor without parameters.
   tnlDistributedGrid();

   //! We do not allow copy constructor without object name.
   tnlDistributedGrid( const tnlDistributedGrid< Dimension, Real, Device, Index >& a );

   public:

   tnlDistributedGrid( const String& name );

   bool init( tnlCommunicator* communicator,
              const GridType& grid,
              const StaticVector< Dimension, Index >& subdomainOverlaps );

   tnlCommunicator< Device >* getCommunicator() const;

   const StaticVector< Dimension, Real >& getDomainLowerCorner() const;

   const StaticVector< Dimension, Real >& getDomainUpperCorner() const;

   const StaticVector< Dimension, Index >& getDimensions() const;

   const StaticVector< Dimension, int >& getGridDimensions() const;

   const StaticVector< Dimension, int >& getLowerNeighbors() const;

   const StaticVector< Dimension, Index >& getLowerSubdomainsOverlaps() const;

   const StaticVector< Dimension, int >& getNodeCoordinates() const;

   const StaticVector< Dimension, Index >& getSubdomainDimensions() const;

   const StaticVector< Dimension, Index >& getUpperSubdomainsOverlaps() const;

   const StaticVector< Dimension, int >& getUppperNeighbors() const;

   protected:

   //! Pointer to the communicator used by this distributed grid.
   tnlCommunicator< Device >* communicator;

   //! In 2D this is the left bottom corner of the global domain.
   /*!***
    * This is naturally generalized to more dimensions.
    */
   StaticVector< Dimension, Real > domainLowerCorner;

   //! In 2D this is the right top corner of the global domain.
   /*!***
    * This is naturally generalized to more dimensions.
    */
   StaticVector< Dimension, Real > domainUpperCorner;

   //! Dimension of the global domain.
   StaticVector< Dimension, Index > globalDimensions;

   //! Dimension of the local subdomain.
   StaticVector< Dimension, Index > subdomainDimensions;

   //! Number of the distributed grid nodes along each dimension.
   StaticVector< Dimension, int > gridDimensions;

   //! Coordinates of this node of the distributed grid.
   StaticVector< Dimension, int > nodeCoordinates;

   //! Here are device IDs taken from the tnlCommunicator.
   /*!***
    * In 2D, this is the device ID of the neighbor on the
    * right and above.
    */
   StaticVector< Dimension, int > uppperNeighbors;

   //! Here are device IDs taken from the tnlCommunicator.
   /*!***
    * In 2D, this is the device ID of the neighbor on the
    * left and below.
    */
   StaticVector< Dimension, int > lowerNeighbors;

   //! Here are widths of overlaps at subdomain boundaries with neighbors.
   /*!***
    * These overlaps are necessary for exchange of data
    * between neighboring nodes. In 2D, here are overlaps
    * with the neighbors on the right and above.
    */
   StaticVector< Dimension, Index > upperSubdomainsOverlaps;

   //! Here are widths of overlaps at subdomain boundaries with neighbors.
   /*!***
    * These overlaps are necessary for exchange of data
    * between neighboring nodes. In 2D, here are overlaps
    * with the neighbors on the left and below.
    */
   StaticVector< Dimension, Index > lowerSubdomainsOverlaps;

};

template< int Dimension, typename GridType, typename Device, typename Real, typename Index >
tnlDistributedGrid< Dimension, GridType, Device, Real, Index > :: tnlDistributedGrid( const String& name )
 : Object( name )
{

}

template< int Dimension, typename GridType, typename Device, typename Real, typename Index >
bool tnlDistributedGrid< Dimension, GridType, Device, Real, Index > :: init( tnlCommunicator* communicator,
                                                                              const GridType& grid,
                                                                              const StaticVector< Dimension, int >& gridDimensions,
                                                                              const StaticVector< Dimension, Index >& subdomainOverlaps )
{

}

template< int Dimension, typename GridType, typename Device, typename Real, typename Index >
tnlCommunicator* tnlDistributedGrid< Dimension, GridType, Device, Real, Index > :: getCommunicator() const
{
    return communicator;
}

template< int Dimension, typename GridType, typename Device, typename Real, typename Index >
const StaticVector< Dimension, Real >& tnlDistributedGrid< Dimension, GridType, Device, Real, Index > :: getDomainLowerCorner() const
{
    return domainLowerCorner;
}

template< int Dimension, typename GridType, typename Device, typename Real, typename Index >
const StaticVector< Dimension, Real >& tnlDistributedGrid< Dimension, GridType, Device, Real, Index > :: getDomainUpperCorner() const
{
    return domainUpperCorner;
}

template< int Dimension, typename GridType, typename Device, typename Real, typename Index >
const StaticVector< Dimension, Index >& tnlDistributedGrid< Dimension, GridType, Device, Real, Index > :: getDimensions() const
{
    return globalDimensions;
}

template< int Dimension, typename GridType, typename Device, typename Real, typename Index >
const StaticVector< Dimension, int >& tnlDistributedGrid< Dimension, GridType, Device, Real, Index > :: getGridDimensions() const
{
    return gridDimensions;
}

template< int Dimension, typename GridType, typename Device, typename Real, typename Index >
const StaticVector< Dimension, int >& tnlDistributedGrid< Dimension, GridType, Device, Real, Index > :: getLowerNeighbors() const
{
    return lowerNeighbors;
}

template< int Dimension, typename GridType, typename Device, typename Real, typename Index >
const StaticVector< Dimension, Index >& tnlDistributedGrid< Dimension, GridType, Device, Real, Index > :: getLowerSubdomainsOverlaps() const
{
    return lowerSubdomainsOverlaps;
}

template< int Dimension, typename GridType, typename Device, typename Real, typename Index >
const StaticVector< Dimension, int >& tnlDistributedGrid< Dimension, GridType, Device, Real, Index > :: getNodeCoordinates() const
{
    return nodeCoordinates;
}

template< int Dimension, typename GridType, typename Device, typename Real, typename Index >
const StaticVector< Dimension, Index >& tnlDistributedGrid< Dimension, GridType, Device, Real, Index > :: getSubdomainDimensions() const
{
    return subdomainDimensions;
}

template< int Dimension, typename GridType, typename Device, typename Real, typename Index >
const StaticVector< Dimension, Index >& tnlDistributedGrid< Dimension, GridType, Device, Real, Index > :: getUpperSubdomainsOverlaps() const
{
    return upperSubdomainsOverlaps;
}

template< int Dimension, typename GridType, typename Device, typename Real, typename Index >
const StaticVector< Dimension, int >& tnlDistributedGrid< Dimension, GridType, Device, Real, Index > :: getUppperNeighbors() const
{
    return uppperNeighbors;
}

#endif /* TNLDISTRIBUTEDGRID_H_ */
