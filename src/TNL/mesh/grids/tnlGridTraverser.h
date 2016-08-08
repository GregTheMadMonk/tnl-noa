/***************************************************************************
                          tnlGridTraverser.h  -  description
                             -------------------
    begin                : Jan 2, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once


namespace TNL {

/****
 * This is only a helper class for tnlTraverser specializations for tnlGrid.
 */
template< typename Grid >
class tnlGridTraverser
{
};

/****
 * 1D grid, Devices::Host
 */
template< typename Real,
          typename Index >
class tnlGridTraverser< tnlGrid< 1, Real, Devices::Host, Index > >
{
   public:
      
      typedef tnlGrid< 1, Real, Devices::Host, Index > GridType;
      typedef SharedPointer< GridType > GridPointer;
      typedef Real RealType;
      typedef Devices::Host DeviceType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
 
      template<
         typename GridEntity,
         typename EntitiesProcessor,
         typename UserData,
         bool processOnlyBoundaryEntities >
      static void
      processEntities(
         const GridPointer& gridPointer,
         const CoordinatesType& begin,
         const CoordinatesType& end,
         const CoordinatesType& entityOrientation,
         const CoordinatesType& entityBasis,
         UserData& userData );
};

/****
 * 1D grid, Devices::Cuda
 */
template< typename Real,
          typename Index >
class tnlGridTraverser< tnlGrid< 1, Real, Devices::Cuda, Index > >
{
   public:
      
      typedef tnlGrid< 1, Real, Devices::Cuda, Index > GridType;
      typedef SharedPointer< GridType > GridPointer;
      typedef Real RealType;
      typedef Devices::Host DeviceType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
 
      template<
         typename GridEntity,
         typename EntitiesProcessor,
         typename UserData,
         bool processOnlyBoundaryEntities  >
      static void
      processEntities(
         const GridPointer& gridPointer,
         const CoordinatesType& begin,
         const CoordinatesType& end,
         const CoordinatesType& entityOrientation,
         const CoordinatesType& entityBasis,
         UserData& userData );
};

/****
 * 2D grid, Devices::Host
 */
template< typename Real,
          typename Index >
class tnlGridTraverser< tnlGrid< 2, Real, Devices::Host, Index > >
{
   public:
      
      typedef tnlGrid< 2, Real, Devices::Host, Index > GridType;
      typedef SharedPointer< GridType > GridPointer;
      typedef Real RealType;
      typedef Devices::Host DeviceType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
 
      template<
         typename GridEntity,
         typename EntitiesProcessor,
         typename UserData,
         bool processOnlyBoundaryEntities,
         int XOrthogonalBoundary = 1,
         int YOrthogonalBoundary = 1 >
      static void
      processEntities(
         const GridPointer& gridPointer,
         const CoordinatesType begin,
         const CoordinatesType end,
         const CoordinatesType& entityOrientation,
         const CoordinatesType& entityBasis,
         UserData& userData );
};

/****
 * 2D grid, Devices::Cuda
 */
template< typename Real,
          typename Index >
class tnlGridTraverser< tnlGrid< 2, Real, Devices::Cuda, Index > >
{
   public:
      
      typedef tnlGrid< 2, Real, Devices::Cuda, Index > GridType;
      typedef SharedPointer< GridType > GridPointer;
      typedef Real RealType;
      typedef Devices::Host DeviceType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
 
      template<
         typename GridEntity,
         typename EntitiesProcessor,
         typename UserData,
         bool processOnlyBoundaryEntities,
         int XOrthogonalBoundary = 1,
         int YOrthogonalBoundary = 1  >
      static void
      processEntities(
         const GridPointer& gridPointer,
         const CoordinatesType& begin,
         const CoordinatesType& end,
         const CoordinatesType& entityOrientation,
         const CoordinatesType& entityBasis,
         UserData& userData );
};

/****
 * 3D grid, Devices::Host
 */
template< typename Real,
          typename Index >
class tnlGridTraverser< tnlGrid< 3, Real, Devices::Host, Index > >
{
   public:
      
      typedef tnlGrid< 3, Real, Devices::Host, Index > GridType;
      typedef SharedPointer< GridType > GridPointer;
      typedef Real RealType;
      typedef Devices::Host DeviceType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
 
      template<
         typename GridEntity,
         typename EntitiesProcessor,
         typename UserData,
         bool processOnlyBoundaryEntities,
         int XOrthogonalBoundary = 1,
         int YOrthogonalBoundary = 1,
         int ZOrthogonalBoundary = 1 >
      static void
      processEntities(
         const GridPointer& gridPointer,
         const CoordinatesType& begin,
         const CoordinatesType& end,
         const CoordinatesType& entityOrientation,
         const CoordinatesType& entityBasis,
         UserData& userData );
};

/****
 * 3D grid, Devices::Cuda
 */
template< typename Real,
          typename Index >
class tnlGridTraverser< tnlGrid< 3, Real, Devices::Cuda, Index > >
{
   public:
      
      typedef tnlGrid< 3, Real, Devices::Cuda, Index > GridType;
      typedef SharedPointer< GridType > GridPointer;
      typedef Real RealType;
      typedef Devices::Host DeviceType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
 
      template<
         typename GridEntity,
         typename EntitiesProcessor,
         typename UserData,
         bool processOnlyBoundaryEntities,
         int XOrthogonalBoundary = 1,
         int YOrthogonalBoundary = 1,
         int ZOrthogonalBoundary = 1 >
      static void
      processEntities(
         const GridPointer& gridPointer,
         const CoordinatesType& begin,
         const CoordinatesType& end,
         const CoordinatesType& entityOrientation,
         const CoordinatesType& entityBasis,
         UserData& userData );
};

} // namespace TNL

#include <TNL/mesh/grids/tnlGridTraverser_impl.h>

