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
 * 1D grid, tnlHost
 */
template< typename Real,
          typename Index >
class tnlGridTraverser< tnlGrid< 1, Real, tnlHost, Index > >
{
   public:
 
      typedef tnlGrid< 1, Real, tnlHost, Index > GridType;
      typedef Real RealType;
      typedef tnlHost DeviceType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
 
      template<
         typename GridEntity,
         typename EntitiesProcessor,
         typename UserData,
         bool processOnlyBoundaryEntities >
      static void
      processEntities(
         const GridType& grid,
         const CoordinatesType& begin,
         const CoordinatesType& end,
         const CoordinatesType& entityOrientation,
         const CoordinatesType& entityBasis,
         UserData& userData );
};

/****
 * 1D grid, tnlCuda
 */
template< typename Real,
          typename Index >
class tnlGridTraverser< tnlGrid< 1, Real, tnlCuda, Index > >
{
   public:
 
      typedef tnlGrid< 1, Real, tnlCuda, Index > GridType;
      typedef Real RealType;
      typedef tnlHost DeviceType;
      typedef Index IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
 
      template<
         typename GridEntity,
         typename EntitiesProcessor,
         typename UserData,
         bool processOnlyBoundaryEntities  >
      static void
      processEntities(
         const GridType& grid,
         const CoordinatesType& begin,
         const CoordinatesType& end,
         const CoordinatesType& entityOrientation,
         const CoordinatesType& entityBasis,
         UserData& userData );
};

/****
 * 2D grid, tnlHost
 */
template< typename Real,
          typename Index >
class tnlGridTraverser< tnlGrid< 2, Real, tnlHost, Index > >
{
   public:
 
      typedef tnlGrid< 2, Real, tnlHost, Index > GridType;
      typedef Real RealType;
      typedef tnlHost DeviceType;
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
         const GridType& grid,
         const CoordinatesType begin,
         const CoordinatesType end,
         const CoordinatesType& entityOrientation,
         const CoordinatesType& entityBasis,
         UserData& userData );
};

/****
 * 2D grid, tnlCuda
 */
template< typename Real,
          typename Index >
class tnlGridTraverser< tnlGrid< 2, Real, tnlCuda, Index > >
{
   public:
 
      typedef tnlGrid< 2, Real, tnlCuda, Index > GridType;
      typedef Real RealType;
      typedef tnlHost DeviceType;
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
         const GridType& grid,
         const CoordinatesType& begin,
         const CoordinatesType& end,
         const CoordinatesType& entityOrientation,
         const CoordinatesType& entityBasis,
         UserData& userData );
};

/****
 * 3D grid, tnlHost
 */
template< typename Real,
          typename Index >
class tnlGridTraverser< tnlGrid< 3, Real, tnlHost, Index > >
{
   public:
 
      typedef tnlGrid< 3, Real, tnlHost, Index > GridType;
      typedef Real RealType;
      typedef tnlHost DeviceType;
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
         const GridType& grid,
         const CoordinatesType& begin,
         const CoordinatesType& end,
         const CoordinatesType& entityOrientation,
         const CoordinatesType& entityBasis,
         UserData& userData );
};

/****
 * 3D grid, tnlCuda
 */
template< typename Real,
          typename Index >
class tnlGridTraverser< tnlGrid< 3, Real, tnlCuda, Index > >
{
   public:
 
      typedef tnlGrid< 3, Real, tnlCuda, Index > GridType;
      typedef Real RealType;
      typedef tnlHost DeviceType;
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
         const GridType& grid,
         const CoordinatesType& begin,
         const CoordinatesType& end,
         const CoordinatesType& entityOrientation,
         const CoordinatesType& entityBasis,
         UserData& userData );
};

} // namespace TNL

#include <mesh/grids/tnlGridTraverser_impl.h>

