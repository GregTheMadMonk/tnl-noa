/***************************************************************************
                          tnlTraverser_Grid3D.h  -  description
                             -------------------
    begin                : Jul 29, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
class tnlTraverser< tnlGrid< 3, Real, Device, Index >, GridEntity, 3 >
{
   public:
      typedef tnlGrid< 3, Real, Device, Index > GridType;
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;

      template< typename UserData,
                typename EntitiesProcessor >
      void processBoundaryEntities( const GridType& grid,
                                    UserData& userData ) const;

      template< typename UserData,
                typename EntitiesProcessor >
      void processInteriorEntities( const GridType& grid,
                                    UserData& userData ) const;
      template< typename UserData,
                typename EntitiesProcessor >
      void processAllEntities( const GridType& grid,
                               UserData& userData ) const;
 
};

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
class tnlTraverser< tnlGrid< 3, Real, Device, Index >, GridEntity, 2 >
{
   public:
      typedef tnlGrid< 3, Real, Device, Index > GridType;
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;

      template< typename UserData,
                typename EntitiesProcessor >
      void processBoundaryEntities( const GridType& grid,
                                    UserData& userData ) const;

      template< typename UserData,
                typename EntitiesProcessor >
      void processInteriorEntities( const GridType& grid,
                                    UserData& userData ) const;
      template< typename UserData,
                typename EntitiesProcessor >
      void processAllEntities( const GridType& grid,
                               UserData& userData ) const;
 
};

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
class tnlTraverser< tnlGrid< 3, Real, Device, Index >, GridEntity, 1 >
{
   public:
      typedef tnlGrid< 3, Real, Device, Index > GridType;
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;

      template< typename UserData,
                typename EntitiesProcessor >
      void processBoundaryEntities( const GridType& grid,
                                    UserData& userData ) const;

      template< typename UserData,
                typename EntitiesProcessor >
      void processInteriorEntities( const GridType& grid,
                                    UserData& userData ) const;

      template< typename UserData,
                typename EntitiesProcessor >
      void processAllEntities( const GridType& grid,
                               UserData& userData ) const;
 
};

template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
class tnlTraverser< tnlGrid< 3, Real, Device, Index >, GridEntity, 0 >
{
   public:
      typedef tnlGrid< 3, Real, Device, Index > GridType;
      typedef Real RealType;
      typedef Device DeviceType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;

      template< typename UserData,
                typename EntitiesProcessor >
      void processBoundaryEntities( const GridType& grid,
                                    UserData& userData ) const;

      template< typename UserData,
                typename EntitiesProcessor >
      void processInteriorEntities( const GridType& grid,
                                    UserData& userData ) const;
 
      template< typename UserData,
                typename EntitiesProcessor >
      void processAllEntities( const GridType& grid,
                               UserData& userData ) const;
};



#ifdef UNDEF


template< typename Real,
          typename Index,
          typename GridEntity >
class tnlTraverser< tnlGrid< 3, Real, Devices::Host, Index >, GridEntity, 3 >
{
   public:
      typedef tnlGrid< 3, Real, Devices::Host, Index > GridType;
      typedef Real RealType;
      typedef Devices::Host DeviceType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;

      template< typename UserData,
                typename EntitiesProcessor >
      void processBoundaryEntities( const GridType& grid,
                                    UserData& userData ) const;

      template< typename UserData,
                typename EntitiesProcessor >
      void processInteriorEntities( const GridType& grid,
                                    UserData& userData ) const;

      template< typename UserData,
                typename EntitiesProcessor >
      void processAllEntities( const GridType& grid,
                               UserData& userData ) const;
 
   protected:

      template< typename UserData,
                typename EntitiesProcessor >
      void processSubgridEntities(
         const GridType& grid,
         const CoordinatesType& begin,
         const CoordinatesType& end,
         GridEntity& entity,
         UserData& userData ) const;
 
};

template< typename Real,
          typename Index,
          typename GridEntity >
class tnlTraverser< tnlGrid< 3, Real, Devices::Host, Index >, GridEntity, 2 >
{
   public:
      typedef tnlGrid< 3, Real, Devices::Host, Index > GridType;
      typedef Real RealType;
      typedef Devices::Host DeviceType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;

      template< typename UserData,
                typename EntitiesProcessor >
      void processBoundaryEntities( const GridType& grid,
                                    UserData& userData ) const;

      template< typename UserData,
                typename EntitiesProcessor >
      void processInteriorEntities( const GridType& grid,
                                    UserData& userData ) const;
 
      template< typename UserData,
                typename EntitiesProcessor >
      void processAllEntities( const GridType& grid,
                               UserData& userData ) const;
 
   protected:

      template< typename UserData,
                typename EntitiesProcessor >
      void processSubgridEntities(
         const GridType& grid,
         const CoordinatesType& begin,
         const CoordinatesType& end,
         GridEntity& entity,
         UserData& userData ) const;
 
};

template< typename Real,
          typename Index,
          typename GridEntity >
class tnlTraverser< tnlGrid< 3, Real, Devices::Host, Index >, GridEntity, 1 >
{
   public:
      typedef tnlGrid< 3, Real, Devices::Host, Index > GridType;
      typedef Real RealType;
      typedef Devices::Host DeviceType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;

      template< typename UserData,
                typename EntitiesProcessor >
      void processBoundaryEntities( const GridType& grid,
                                    UserData& userData ) const;

      template< typename UserData,
                typename EntitiesProcessor >
      void processInteriorEntities( const GridType& grid,
                                    UserData& userData ) const;
 
      template< typename UserData,
                typename EntitiesProcessor >
      void processAllEntities( const GridType& grid,
                               UserData& userData ) const;
 
   protected:

      template< typename UserData,
                typename EntitiesProcessor >
      void processSubgridEntities(
         const GridType& grid,
         const CoordinatesType& begin,
         const CoordinatesType& end,
         GridEntity& entity,
         UserData& userData ) const;
};

template< typename Real,
          typename Index,
          typename GridEntity >
class tnlTraverser< tnlGrid< 3, Real, Devices::Host, Index >, GridEntity, 0 >
{
   public:
      typedef tnlGrid< 3, Real, Devices::Host, Index > GridType;
      typedef Real RealType;
      typedef Devices::Host DeviceType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;

      template< typename UserData,
                typename EntitiesProcessor >
      void processBoundaryEntities( const GridType& grid,
                                    UserData& userData ) const;

      template< typename UserData,
                typename EntitiesProcessor >
      void processInteriorEntities( const GridType& grid,
                                    UserData& userData ) const;
 
      template< typename UserData,
                typename EntitiesProcessor >
      void processAllEntities( const GridType& grid,
                               UserData& userData ) const;
 
   protected:

      template< typename UserData,
                typename EntitiesProcessor >
      void processSubgridEntities(
         const GridType& grid,
         const CoordinatesType& begin,
         const CoordinatesType& end,
         GridEntity& entity,
         UserData& userData ) const;
};

/****
 * CUDA traversal
 */

template< typename Real,
          typename Index,
          typename GridEntity >
class tnlTraverser< tnlGrid< 3, Real, Devices::Cuda, Index >, GridEntity, 3 >
{
   public:
      typedef tnlGrid< 3, Real, Devices::Cuda, Index > GridType;
      typedef Real RealType;
      typedef Devices::Cuda DeviceType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;

      template< typename UserData,
                typename EntitiesProcessor >
      void processBoundaryEntities( const GridType& grid,
                                    UserData& userData ) const;

      template< typename UserData,
                typename EntitiesProcessor >
      void processInteriorEntities( const GridType& grid,
                                    UserData& userData ) const;
 
      template< typename UserData,
                typename EntitiesProcessor >
      void processAllEntities( const GridType& grid,
                               UserData& userData ) const;
 
   protected:

      template< typename UserData,
                typename EntitiesProcessor >
      void processSubgridEntities(
         const GridType& grid,
         const CoordinatesType& begin,
         const CoordinatesType& end,
         GridEntity& entity,
         UserData& userData ) const;
};

template< typename Real,
          typename Index,
          typename GridEntity >
class tnlTraverser< tnlGrid< 3, Real, Devices::Cuda, Index >, GridEntity, 2 >
{
   public:
      typedef tnlGrid< 3, Real, Devices::Cuda, Index > GridType;
      typedef Real RealType;
      typedef Devices::Cuda DeviceType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;

      template< typename UserData,
                typename EntitiesProcessor >
      void processBoundaryEntities( const GridType& grid,
                                    UserData& userData ) const;

      template< typename UserData,
                typename EntitiesProcessor >
      void processInteriorEntities( const GridType& grid,
                                    UserData& userData ) const;
 
      template< typename UserData,
                typename EntitiesProcessor >
      void processAllEntities( const GridType& grid,
                               UserData& userData ) const;
 
   protected:

      template< typename UserData,
                typename EntitiesProcessor >
      void processSubgridEntities(
         const GridType& grid,
         const CoordinatesType& begin,
         const CoordinatesType& end,
         GridEntity& entity,
         UserData& userData ) const;
};

template< typename Real,
          typename Index,
          typename GridEntity >
class tnlTraverser< tnlGrid< 3, Real, Devices::Cuda, Index >, GridEntity, 1 >
{
   public:
      typedef tnlGrid< 3, Real, Devices::Cuda, Index > GridType;
      typedef Real RealType;
      typedef Devices::Cuda DeviceType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;

      template< typename UserData,
                typename EntitiesProcessor >
      void processBoundaryEntities( const GridType& grid,
                                    UserData& userData ) const;

      template< typename UserData,
                typename EntitiesProcessor >
      void processInteriorEntities( const GridType& grid,
                                    UserData& userData ) const;
 
      template< typename UserData,
                typename EntitiesProcessor >
      void processAllEntities( const GridType& grid,
                               UserData& userData ) const;
 
   protected:

      template< typename UserData,
                typename EntitiesProcessor >
      void processSubgridEntities(
         const GridType& grid,
         const CoordinatesType& begin,
         const CoordinatesType& end,
         GridEntity& entity,
         UserData& userData ) const;
 
};

template< typename Real,
          typename Index,
          typename GridEntity >
class tnlTraverser< tnlGrid< 3, Real, Devices::Cuda, Index >, GridEntity, 0 >
{
   public:
      typedef tnlGrid< 3, Real, Devices::Cuda, Index > GridType;
      typedef Real RealType;
      typedef Devices::Cuda DeviceType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;

      template< typename UserData,
                typename EntitiesProcessor >
      void processBoundaryEntities( const GridType& grid,
                                    UserData& userData ) const;

      template< typename UserData,
                typename EntitiesProcessor >
      void processInteriorEntities( const GridType& grid,
                                    UserData& userData ) const;
 
      template< typename UserData,
                typename EntitiesProcessor >
      void processAllEntities( const GridType& grid,
                               UserData& userData ) const;
 
   protected:

      template< typename UserData,
                typename EntitiesProcessor >
      void processSubgridEntities(
         const GridType& grid,
         const CoordinatesType& begin,
         const CoordinatesType& end,
         GridEntity& entity,
         UserData& userData ) const;
 
};
#endif // UNDEF

} // namespace TNL

#include <TNL/mesh/grids/tnlTraverser_Grid3D_impl.h>
