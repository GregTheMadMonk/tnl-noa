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

#include <core/tnlObject.h>
#include <core/tnlCommunicator.h>

template< int Dimensions,
          typename GridType,
          typename Device = tnlHost,
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
              const tnlStaticVector< Dimensions, Index >& subdomainOverlaps );

   tnlCommunicator< Device >* getCommunicator() const;

   const tnlStaticVector< Dimensions, Real >& getDomainLowerCorner() const;

   const tnlStaticVector< Dimensions, Real >& getDomainUpperCorner() const;

   const tnlStaticVector< Dimensions, Index >& getDimensions() const;

   const tnlStaticVector< Dimensions, int >& getGridDimensions() const;

   const tnlStaticVector< Dimensions, int >& getLowerNeighbors() const;

   const tnlStaticVector< Dimensions, Index >& getLowerSubdomainsOverlaps() const;

   const tnlStaticVector< Dimensions, int >& getNodeCoordinates() const;

   const tnlStaticVector< Dimensions, Index >& getSubdomainDimensions() const;

   const tnlStaticVector< Dimensions, Index >& getUpperSubdomainsOverlaps() const;

   const tnlStaticVector< Dimensions, int >& getUppperNeighbors() const;

   protected:

   //! Pointer to the communicator used by this distributed grid.
   tnlCommunicator< Device >* communicator;

   //! In 2D this is the left bottom corner of the global domain.
   /*!***
    * This is naturally generalized to more dimensions.
    */
   tnlStaticVector< Dimensions, Real > domainLowerCorner;

   //! In 2D this is the right top corner of the global domain.
   /*!***
    * This is naturally generalized to more dimensions.
    */
   tnlStaticVector< Dimensions, Real > domainUpperCorner;

   //! Dimensions of the global domain.
   tnlStaticVector< Dimensions, Index > globalDimensions;

   //! Dimensions of the local subdomain.
   tnlStaticVector< Dimensions, Index > subdomainDimensions;

   //! Number of the distributed grid nodes along each dimension.
   tnlStaticVector< Dimensions, int > gridDimensions;

   //! Coordinates of this node of the distributed grid.
   tnlStaticVector< Dimensions, int > nodeCoordinates;

   //! Here are device IDs taken from the tnlCommunicator.
   /*!***
    * In 2D, this is the device ID of the neighbor on the
    * right and above.
    */
   tnlStaticVector< Dimensions, int > uppperNeighbors;

   //! Here are device IDs taken from the tnlCommunicator.
   /*!***
    * In 2D, this is the device ID of the neighbor on the
    * left and below.
    */
   tnlStaticVector< Dimensions, int > lowerNeighbors;

   //! Here are widths of overlaps at subdomain boundaries with neighbors.
   /*!***
    * These overlaps are necessary for exchange of data
    * between neighboring nodes. In 2D, here are overlaps
    * with the neighbors on the right and above.
    */
   tnlStaticVector< Dimensions, Index > upperSubdomainsOverlaps;

   //! Here are widths of overlaps at subdomain boundaries with neighbors.
   /*!***
    * These overlaps are necessary for exchange of data
    * between neighboring nodes. In 2D, here are overlaps
    * with the neighbors on the left and below.
    */
   tnlStaticVector< Dimensions, Index > lowerSubdomainsOverlaps;

};

template< int Dimensions, typename GridType, typename Device, typename Real, typename Index >
tnlDistributedGrid< Dimensions, GridType, Device, Real, Index > :: tnlDistributedGrid( const tnlString& name )
 : tnlObject( name )
{

}

template< int Dimensions, typename GridType, typename Device, typename Real, typename Index >
bool tnlDistributedGrid< Dimensions, GridType, Device, Real, Index > :: init( tnlCommunicator* communicator,
                                                                              const GridType& grid,
                                                                              const tnlStaticVector< Dimensions, int >& gridDimensions,
                                                                              const tnlStaticVector< Dimensions, Index >& subdomainOverlaps )
{

}

template< int Dimensions, typename GridType, typename Device, typename Real, typename Index >
tnlCommunicator* tnlDistributedGrid< Dimensions, GridType, Device, Real, Index > :: getCommunicator() const
{
    return communicator;
}

template< int Dimensions, typename GridType, typename Device, typename Real, typename Index >
const tnlStaticVector< Dimensions, Real >& tnlDistributedGrid< Dimensions, GridType, Device, Real, Index > :: getDomainLowerCorner() const
{
    return domainLowerCorner;
}

template< int Dimensions, typename GridType, typename Device, typename Real, typename Index >
const tnlStaticVector< Dimensions, Real >& tnlDistributedGrid< Dimensions, GridType, Device, Real, Index > :: getDomainUpperCorner() const
{
    return domainUpperCorner;
}

template< int Dimensions, typename GridType, typename Device, typename Real, typename Index >
const tnlStaticVector< Dimensions, Index >& tnlDistributedGrid< Dimensions, GridType, Device, Real, Index > :: getDimensions() const
{
    return globalDimensions;
}

template< int Dimensions, typename GridType, typename Device, typename Real, typename Index >
const tnlStaticVector< Dimensions, int >& tnlDistributedGrid< Dimensions, GridType, Device, Real, Index > :: getGridDimensions() const
{
    return gridDimensions;
}

template< int Dimensions, typename GridType, typename Device, typename Real, typename Index >
const tnlStaticVector< Dimensions, int >& tnlDistributedGrid< Dimensions, GridType, Device, Real, Index > :: getLowerNeighbors() const
{
    return lowerNeighbors;
}

template< int Dimensions, typename GridType, typename Device, typename Real, typename Index >
const tnlStaticVector< Dimensions, Index >& tnlDistributedGrid< Dimensions, GridType, Device, Real, Index > :: getLowerSubdomainsOverlaps() const
{
    return lowerSubdomainsOverlaps;
}

template< int Dimensions, typename GridType, typename Device, typename Real, typename Index >
const tnlStaticVector< Dimensions, int >& tnlDistributedGrid< Dimensions, GridType, Device, Real, Index > :: getNodeCoordinates() const
{
    return nodeCoordinates;
}

template< int Dimensions, typename GridType, typename Device, typename Real, typename Index >
const tnlStaticVector< Dimensions, Index >& tnlDistributedGrid< Dimensions, GridType, Device, Real, Index > :: getSubdomainDimensions() const
{
    return subdomainDimensions;
}

template< int Dimensions, typename GridType, typename Device, typename Real, typename Index >
const tnlStaticVector< Dimensions, Index >& tnlDistributedGrid< Dimensions, GridType, Device, Real, Index > :: getUpperSubdomainsOverlaps() const
{
    return upperSubdomainsOverlaps;
}

template< int Dimensions, typename GridType, typename Device, typename Real, typename Index >
const tnlStaticVector< Dimensions, int >& tnlDistributedGrid< Dimensions, GridType, Device, Real, Index > :: getUppperNeighbors() const
{
    return uppperNeighbors;
}

#endif /* TNLDISTRIBUTEDGRID_H_ */
