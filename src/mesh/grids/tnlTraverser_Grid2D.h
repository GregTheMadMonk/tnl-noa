/***************************************************************************
                          tnlTraverser_Grid2D.h  -  description
                             -------------------
    begin                : Jul 29, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */


#ifndef TNLTRAVERSER_GRID2D_H_
#define TNLTRAVERSER_GRID2D_H_


template< typename Real,
          typename Device,
          typename Index,
          typename GridEntity >
class tnlTraverser< tnlGrid< 2, Real, Device, Index >, GridEntity, 2 >
{
   public:
      typedef tnlGrid< 2, Real, Device, Index > GridType;
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
class tnlTraverser< tnlGrid< 2, Real, Device, Index >, GridEntity, 1 >
{
   public:
      typedef tnlGrid< 2, Real, Device, Index > GridType;
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
class tnlTraverser< tnlGrid< 2, Real, Device, Index >, GridEntity, 0 >
{
   public:
      typedef tnlGrid< 2, Real, Device, Index > GridType;
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
class tnlTraverser< tnlGrid< 2, Real, tnlCuda, Index >, GridEntity, 2 >
{
   public:
      typedef tnlGrid< 2, Real, tnlCuda, Index > GridType;
      typedef Real RealType;
      typedef tnlCuda DeviceType;
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
class tnlTraverser< tnlGrid< 2, Real, tnlCuda, Index >, GridEntity, 1 >
{
   public:
      typedef tnlGrid< 2, Real, tnlCuda, Index > GridType;
      typedef Real RealType;
      typedef tnlCuda DeviceType;
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
class tnlTraverser< tnlGrid< 2, Real, tnlCuda, Index >, GridEntity, 0 >
{
   public:
      typedef tnlGrid< 2, Real, tnlCuda, Index > GridType;
      typedef Real RealType;
      typedef tnlCuda DeviceType;
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

#include <mesh/grids/tnlTraverser_Grid2D_impl.h>

#endif /* TNLTRAVERSER_GRID2D_H_ */
