/***************************************************************************
                          GridTraverser.h  -  description
                             -------------------
    begin                : Jan 2, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once


namespace TNL {
namespace Meshes {

/****
 * This is only a helper class for Traverser specializations for Grid.
 */
template< typename Grid >
class GridTraverser
{
};

/****
 * 1D grid, Devices::Host
 */
template< typename Real,
          typename Index >
class GridTraverser< Meshes::Grid< 1, Real, Devices::Host, Index > >
{
   public:
      
      typedef Meshes::Grid< 1, Real, Devices::Host, Index > GridType;
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
         const CoordinatesType begin,
         const CoordinatesType end,
         const CoordinatesType& entityOrientation,
         const CoordinatesType& entityBasis,
         UserData& userData );
};

/****
 * 1D grid, Devices::Cuda
 */
template< typename Real,
          typename Index >
class GridTraverser< Meshes::Grid< 1, Real, Devices::Cuda, Index > >
{
   public:
      
      typedef Meshes::Grid< 1, Real, Devices::Cuda, Index > GridType;
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
class GridTraverser< Meshes::Grid< 2, Real, Devices::Host, Index > >
{
   public:
      
      typedef Meshes::Grid< 2, Real, Devices::Host, Index > GridType;
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
class GridTraverser< Meshes::Grid< 2, Real, Devices::Cuda, Index > >
{
   public:
      
      typedef Meshes::Grid< 2, Real, Devices::Cuda, Index > GridType;
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
class GridTraverser< Meshes::Grid< 3, Real, Devices::Host, Index > >
{
   public:
      
      typedef Meshes::Grid< 3, Real, Devices::Host, Index > GridType;
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
         const CoordinatesType begin,
         const CoordinatesType end,
         const CoordinatesType& entityOrientation,
         const CoordinatesType& entityBasis,
         UserData& userData );
};

/****
 * 3D grid, Devices::Cuda
 */
template< typename Real,
          typename Index >
class GridTraverser< Meshes::Grid< 3, Real, Devices::Cuda, Index > >
{
   public:
      
      typedef Meshes::Grid< 3, Real, Devices::Cuda, Index > GridType;
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

} // namespace Meshes
} // namespace TNL

#include <TNL/Meshes/GridDetails/GridTraverser_impl.h>

