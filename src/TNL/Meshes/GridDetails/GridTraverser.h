/***************************************************************************
                          GridTraverser.h  -  description
                             -------------------
    begin                : Jan 2, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Meshes/Grid.h>
#include <TNL/SharedPointer.h>
#include <TNL/CudaStreamPool.h>

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
         SharedPointer< UserData, DeviceType >& userData,
         const int& stream = 0 );
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
      typedef Devices::Cuda DeviceType;
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
         SharedPointer< UserData, DeviceType >& userData,
         const int& stream = 0 );
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
         int YOrthogonalBoundary = 1,
         typename... GridEntityParameters >
      static void
      processEntities(
         const GridPointer& gridPointer,
         const CoordinatesType begin,
         const CoordinatesType end,
         SharedPointer< UserData, DeviceType >& userData,
         const int& stream = 0,
         // gridEntityParameters are passed to GridEntity's constructor
         // (i.e. orientation and basis for faces)
         const GridEntityParameters&... gridEntityParameters );
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
      typedef Devices::Cuda DeviceType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
 
      template<
         typename GridEntity,
         typename EntitiesProcessor,
         typename UserData,
         bool processOnlyBoundaryEntities,
         int XOrthogonalBoundary = 1,
         int YOrthogonalBoundary = 1,
         typename... GridEntityParameters >
      static void
      processEntities(
         const GridPointer& gridPointer,
         const CoordinatesType& begin,
         const CoordinatesType& end,
         SharedPointer< UserData, DeviceType >& userData,
         const int& stream = 0,
         // gridEntityParameters are passed to GridEntity's constructor
         // (i.e. orientation and basis for faces)
         const GridEntityParameters&... gridEntityParameters );
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
         int ZOrthogonalBoundary = 1,
         typename... GridEntityParameters >
      static void
      processEntities(
         const GridPointer& gridPointer,
         const CoordinatesType begin,
         const CoordinatesType end,
         SharedPointer< UserData, DeviceType >& userData,
         const int& stream = 0,
         // gridEntityParameters are passed to GridEntity's constructor
         // (i.e. orientation and basis for faces and edges)
         const GridEntityParameters&... gridEntityParameters );
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
      typedef Devices::Cuda DeviceType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
 
      template<
         typename GridEntity,
         typename EntitiesProcessor,
         typename UserData,
         bool processOnlyBoundaryEntities,
         int XOrthogonalBoundary = 1,
         int YOrthogonalBoundary = 1,
         int ZOrthogonalBoundary = 1,
         typename... GridEntityParameters >
      static void
      processEntities(
         const GridPointer& gridPointer,
         const CoordinatesType& begin,
         const CoordinatesType& end,
         SharedPointer< UserData, DeviceType >& userData,
         const int& stream = 0,
         // gridEntityParameters are passed to GridEntity's constructor
         // (i.e. orientation and basis for faces and edges)
         const GridEntityParameters&... gridEntityParameters );
};

} // namespace Meshes
} // namespace TNL

#include <TNL/Meshes/GridDetails/GridTraverser_impl.h>

