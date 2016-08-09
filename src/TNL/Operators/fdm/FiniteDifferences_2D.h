/***************************************************************************
                          FiniteDifferences_2D.h  -  description
                             -------------------
    begin                : Jan 10, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {
namespace Operators {   

/***
 * Default implementation for case when one differentiate with respect
 * to z. In this case the result is zero.
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index,
          int XDifference,
          int YDifference,
          int ZDifference,
          int XDirection,
          int YDirection,
          int ZDirection >
class FiniteDifferences<
   tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index,
   XDifference, YDifference, ZDifference,
   XDirection, YDirection, ZDirection >
{
   static_assert( ZDifference != 0,
      "You try to use default finite difference with 'wrong' template parameters. It means that required finite difference was not implmented yet." );
   public:
 
      template< typename MeshFunction, typename MeshEntity >
      __cuda_callable__
      static Real getValue( const MeshFunction& u,
                            const MeshEntity& entity )
      {
         return 0.0;
      }
};

/****
 * 1st order forward difference
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class FiniteDifferences<
   tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index,
   1, 0, 0,
   1, 0, 0 >
{
   public:
 
      template< typename MeshFunction, typename MeshEntity >
      __cuda_callable__
      static Real getValue( const MeshFunction& u,
                            const MeshEntity& entity )
      {
         const typename MeshEntity::template NeighbourEntities< 2 >& neighbourEntities = entity.getNeighbourEntities();
         const Real& hxDiv = entity.getMesh().template getSpaceStepsProducts< -1, 0 >();
         const Real& u_c = u[ entity.getIndex() ];
         return ( u[ neighbourEntities.template getEntityIndex< 1, 0 >()] - u_c ) * hxDiv;
      }
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class FiniteDifferences<
   tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index,
   0, 1, 0,
   0, 1, 0 >
{
   public:
 
      template< typename MeshFunction, typename MeshEntity >
      __cuda_callable__
      static Real getValue( const MeshFunction& u,
                            const MeshEntity& entity )
      {
         const typename MeshEntity::template NeighbourEntities< 2 >& neighbourEntities = entity.getNeighbourEntities();
         const Real& hyDiv = entity.getMesh().template getSpaceStepsProducts< 0, -1 >();
         const Real& u_c = u[ entity.getIndex() ];
         return ( u[ neighbourEntities.template getEntityIndex< 0, 1 >()] - u_c ) * hyDiv;
      }
};

/****
 * 1st order backward difference
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class FiniteDifferences<
   tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index,
   1, 0, 0,
   -1, 0, 0 >
{
   public:
 
      template< typename MeshFunction, typename MeshEntity >
      __cuda_callable__
      static Real getValue( const MeshFunction& u,
                            const MeshEntity& entity )
      {
         const typename MeshEntity::template NeighbourEntities< 2 >& neighbourEntities = entity.getNeighbourEntities();
         const Real& hxDiv = entity.getMesh().template getSpaceStepsProducts< -1,  0 >();
         const Real& u_c = u[ entity.getIndex() ];
         return ( u_c - u[ neighbourEntities.template getEntityIndex< -1, 0 >()] ) * hxDiv;
      }
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class FiniteDifferences<
   tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index,
   0,  1, 0,
   0, -1, 0 >
{
   public:
 
      template< typename MeshFunction, typename MeshEntity >
      __cuda_callable__
      static Real getValue( const MeshFunction& u,
                            const MeshEntity& entity )
      {
         const typename MeshEntity::template NeighbourEntities< 2 >& neighbourEntities = entity.getNeighbourEntities();
         const Real& hyDiv = entity.getMesh().template getSpaceStepsProducts< 0, -1 >();
         const Real& u_c = u[ entity.getIndex() ];
         return ( u_c - u[ neighbourEntities.template getEntityIndex< 0, -1 >()] ) * hyDiv;
      }
};

/****
 * 1st order central difference
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class FiniteDifferences<
   tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index,
   1, 0, 0,
   0, 0, 0 >
{
   public:
 
      template< typename MeshFunction, typename MeshEntity >
      __cuda_callable__
      static Real getValue( const MeshFunction& u,
                            const MeshEntity& entity )
      {
         const typename MeshEntity::template NeighbourEntities< 2 >& neighbourEntities = entity.getNeighbourEntities();
         const Real& hxDiv = entity.getMesh().template getSpaceStepsProducts< -1, 0 >();
         return ( u[ neighbourEntities.template getEntityIndex< 1, 0 >() ] -
                  u[ neighbourEntities.template getEntityIndex< -1, 0 >() ] ) * ( 0.5 * hxDiv );
      }
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class FiniteDifferences<
   tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index,
   0, 1, 0,
   0, 0, 0 >
{
   public:
 
      template< typename MeshFunction, typename MeshEntity >
      __cuda_callable__
      static Real getValue( const MeshFunction& u,
                            const MeshEntity& entity )
      {
         const typename MeshEntity::template NeighbourEntities< 2 >& neighbourEntities = entity.getNeighbourEntities();
         const Real& hyDiv = entity.getMesh().template getSpaceStepsProducts< 0, -1 >();
         return ( u[ neighbourEntities.template getEntityIndex< 0,  1 >() ] -
                  u[ neighbourEntities.template getEntityIndex< 0, -1 >() ] ) * ( 0.5 * hyDiv );
      }
};


/****
 * 2nd order central difference
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class FiniteDifferences<
   tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index,
   2, 0, 0,
   1, 0, 0 >
{
   public:
 
      template< typename MeshFunction, typename MeshEntity >
      __cuda_callable__
      static Real getValue( const MeshFunction& u,
                            const MeshEntity& entity )
      {
         const typename MeshEntity::template NeighbourEntities< 2 >& neighbourEntities = entity.getNeighbourEntities();
         const Real& hxSquareDiv = entity.getMesh().template getSpaceStepsProducts< -2,0 >();
         const Real& u_c = u[ entity.getIndex() ];
         return ( u[ neighbourEntities.template getEntityIndex< 2, 0 >() ] -
                  2.0 * u_c +
                  u[ neighbourEntities.template getEntityIndex< 1, 0 >() ] ) * hxSquareDiv;
      }
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class FiniteDifferences<
   tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index,
   2, 0, 0,
   -1, 0, 0 >
{
   public:
 
      template< typename MeshFunction, typename MeshEntity >
      __cuda_callable__
      static Real getValue( const MeshFunction& u,
                            const MeshEntity& entity )
      {
         const typename MeshEntity::template NeighbourEntities< 2 >& neighbourEntities = entity.getNeighbourEntities();
         const Real& hxSquareDiv = entity.getMesh().template getSpaceStepsProducts< -2, 0 >();
         const Real& u_c = u[ entity.getIndex() ];
         return ( u[ neighbourEntities.template getEntityIndex< -2, 0 >() ] -
                  2.0 * u_c +
                  u[ neighbourEntities.template getEntityIndex< -1, 0 >() ] ) * hxSquareDiv;
      }
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class FiniteDifferences<
   tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index,
   2, 0, 0,
   0, 0, 0 >
{
   public:
 
      template< typename MeshFunction, typename MeshEntity >
      __cuda_callable__
      static Real getValue( const MeshFunction& u,
                            const MeshEntity& entity )
      {
         const typename MeshEntity::template NeighbourEntities< 2 >& neighbourEntities = entity.getNeighbourEntities();
         const Real& hxSquareDiv = entity.getMesh().template getSpaceStepsProducts< -2, 0 >();
         const Real& u_c = u[ entity.getIndex() ];
         return ( u[ neighbourEntities.template getEntityIndex<  1, 0 >() ] -
                  2.0 * u_c +
                  u[ neighbourEntities.template getEntityIndex< -1, 0 >() ] ) * hxSquareDiv;
      }
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class FiniteDifferences<
   tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index,
   0, 2, 0,
   0 ,1, 0 >
{
   public:
 
      template< typename MeshFunction, typename MeshEntity >
      __cuda_callable__
      static Real getValue( const MeshFunction& u,
                            const MeshEntity& entity )
      {
         const typename MeshEntity::template NeighbourEntities< 2 >& neighbourEntities = entity.getNeighbourEntities();
         const Real& hxSquareDiv = entity.getMesh().template getSpaceStepsProducts< 0, -2 >();
         const Real& u_c = u[ entity.getIndex() ];
         return ( u[ neighbourEntities.template getEntityIndex< 0, 2 >() ] -
                  2.0 * u_c +
                  u[ neighbourEntities.template getEntityIndex< 0, 1 >() ] ) * hxSquareDiv;
      }
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class FiniteDifferences<
   tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index,
   0,  2, 0,
   0, -1, 0 >
{
   public:
 
      template< typename MeshFunction, typename MeshEntity >
      __cuda_callable__
      static Real getValue( const MeshFunction& u,
                            const MeshEntity& entity )
      {
         const typename MeshEntity::template NeighbourEntities< 2 >& neighbourEntities = entity.getNeighbourEntities();
         const Real& hxSquareDiv = entity.getMesh().template getSpaceStepsProducts< 0, -2 >();
         const Real& u_c = u[ entity.getIndex() ];
         return ( u[ neighbourEntities.template getEntityIndex< 0, -2 >() ] -
                  2.0 * u_c +
                  u[ neighbourEntities.template getEntityIndex< 0, -1 >() ] ) * hxSquareDiv;
      }
};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class FiniteDifferences<
   tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index,
   0, 2, 0,
   0, 0, 0 >
{
   public:
 
      template< typename MeshFunction, typename MeshEntity >
      __cuda_callable__
      static Real getValue( const MeshFunction& u,
                            const MeshEntity& entity )
      {
         const typename MeshEntity::template NeighbourEntities< 2 >& neighbourEntities = entity.getNeighbourEntities();
         const Real& hySquareDiv = entity.getMesh().template getSpaceStepsProducts< 0, -2 >();
         const Real& u_c = u[ entity.getIndex() ];
         return ( u[ neighbourEntities.template getEntityIndex< 0,  1 >() ] -
                  2.0 * u_c +
                  u[ neighbourEntities.template getEntityIndex< 0, -1 >() ] ) * hySquareDiv;
      }
};

} // namespace Operators
} // namespace TNL