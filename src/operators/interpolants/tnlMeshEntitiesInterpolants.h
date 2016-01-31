/***************************************************************************
                          tnlMeshEntitiesInterpolants.h  -  description
                             -------------------
    begin                : Jan 25, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TNLMESHENTITIESINTERPOLANTS_H
#define	TNLMESHENTITIESINTERPOLANTS_H

#include <type_traits>
#include<functions/tnlDomain.h>

template< typename Mesh,
          int InEntityDimensions,
          int OutEntityDimenions >
class tnlMeshEntitiesInterpolants
{   
};

/***
 * 1D grid mesh entity interpolation: 1 -> 0 
 */
template< typename Real,
          typename Device,
          typename Index >
class tnlMeshEntitiesInterpolants< tnlGrid< 1, Real, Device, Index >, 1, 0 >
   : public tnlDomain< 1, MeshInteriorDomain >
{
   public:
      
      typedef tnlGrid< 1, Real, Device, Index > MeshType;

      template< typename MeshFunction, typename MeshEntity >
      __cuda_callable__
      Real operator()( const MeshFunction& u,
                       const MeshEntity& entity,
                       const Real& time = 0.0 ) const
      {
         static_assert( MeshFunction::getEntityDimensions() == 1,
            "Mesh function must be defined on cells." );

         static_assert( std::is_same< typename MeshEntity::MeshType, MeshType >::value,
            "The mesh entity belongs to other mesh type then the interpolants." );
         
         const typename MeshEntity::template NeighbourEntities< 1 >& neighbourEntities = entity.getNeighbourEntities();      
         
         return 0.5 * ( u[ neighbourEntities.template getEntityIndex< -1 >() ] + 
                        u[ neighbourEntities.template getEntityIndex<  1 >() ] );
      }
};

/***
 * 1D grid mesh entity interpolation: 0 -> 1 
 */
template< typename Real,
          typename Device,
          typename Index >
class tnlMeshEntitiesInterpolants< tnlGrid< 1, Real, Device, Index >, 0, 1 >
   : public tnlDomain< 1, MeshInteriorDomain >
{
   public:
      
      typedef tnlGrid< 1, Real, Device, Index > MeshType;

      template< typename MeshFunction, typename MeshEntity >
      __cuda_callable__
      Real operator()( const MeshFunction& u,
                       const MeshEntity& entity,
                       const Real& time = 0.0 ) const
      {
         static_assert( MeshFunction::getEntityDimensions() == 0,
            "Mesh function must be defined on vertices (or faces in case on 1D grid)." );
         
         static_assert( std::is_same< typename MeshEntity::MeshType, MeshType >::value,
            "The mesh entity belongs to other mesh type then the interpolants." );         
         
         const typename MeshEntity::template NeighbourEntities< 0 >& neighbourEntities = entity.getNeighbourEntities();      
         
         return 0.5 * ( u[ neighbourEntities.template getEntityIndex< -1 >() ] + 
                        u[ neighbourEntities.template getEntityIndex<  1 >() ] );
      }
};

/***
 * 2D grid mesh entity interpolation: 2 -> 1 
 */
template< typename Real,
          typename Device,
          typename Index >
class tnlMeshEntitiesInterpolants< tnlGrid< 2, Real, Device, Index >, 2, 1 >
   : public tnlDomain< 2, MeshInteriorDomain >
{
   public:
      
      typedef tnlGrid< 2, Real, Device, Index > MeshType;
      
      template< typename MeshFunction, typename MeshEntity >
      __cuda_callable__
      Real operator()( const MeshFunction& u,
                       const MeshEntity& entity,
                       const Real& time = 0.0 ) const
      {
         static_assert( MeshFunction::getEntityDimensions() == 2,
            "Mesh function must be defined on cells." );
         
         static_assert( std::is_same< typename MeshEntity::MeshType, MeshType >::value,
            "The mesh entity belongs to other mesh type then the interpolants." );         
         
         const typename MeshEntity::template NeighbourEntities< 2 >& neighbourEntities = entity.getNeighbourEntities();      
         
         if( entity.getOrientation().x() == 1.0 )
            return 0.5 * ( u[ neighbourEntities.template getEntityIndex< -1, 0 >() ] + 
                           u[ neighbourEntities.template getEntityIndex<  1, 0 >() ] );
         else
            return 0.5 * ( u[ neighbourEntities.template getEntityIndex< 0, -1 >() ] + 
                           u[ neighbourEntities.template getEntityIndex< 0,  1 >() ] );
      }
};

/***
 * 2D grid mesh entity interpolation: 2 -> 0 
 */
template< typename Real,
          typename Device,
          typename Index >
class tnlMeshEntitiesInterpolants< tnlGrid< 2, Real, Device, Index >, 2, 0 >
   : public tnlDomain< 2, MeshInteriorDomain >
{
   public:
      
      typedef tnlGrid< 2, Real, Device, Index > MeshType;

      template< typename MeshFunction, typename MeshEntity >
      __cuda_callable__
      Real operator()( const MeshFunction& u,
                       const MeshEntity& entity,
                       const Real& time = 0.0 ) const
      {
         static_assert( MeshFunction::getEntityDimensions() == 2,
            "Mesh function must be defined on cells." );
         
         static_assert( std::is_same< typename MeshEntity::MeshType, MeshType >::value,
            "The mesh entity belongs to other mesh type then the interpolants." );         
         
         const typename MeshEntity::template NeighbourEntities< 2 >& neighbourEntities = entity.getNeighbourEntities();      
                  
         return 0.25 * ( u[ neighbourEntities.template getEntityIndex< -1,  1 >() ] + 
                         u[ neighbourEntities.template getEntityIndex<  1,  1 >() ] +
                         u[ neighbourEntities.template getEntityIndex< -1, -1 >() ] + 
                         u[ neighbourEntities.template getEntityIndex<  1, -1 >() ] );
      }
};

/***
 * 2D grid mesh entity interpolation: 1 -> 2 
 */
template< typename Real,
          typename Device,
          typename Index >
class tnlMeshEntitiesInterpolants< tnlGrid< 2, Real, Device, Index >, 1, 2 >
   : public tnlDomain< 2, MeshInteriorDomain >
{
   public:
      
      typedef tnlGrid< 2, Real, Device, Index > MeshType;

      template< typename MeshFunction, typename MeshEntity >
      __cuda_callable__
      Real operator()( const MeshFunction& u,
                       const MeshEntity& entity,
                       const Real& time = 0.0 ) const
      {
         static_assert( MeshFunction::getEntityDimensions() == 1,
            "Mesh function must be defined on faces." );
         
         static_assert( std::is_same< typename MeshEntity::MeshType, MeshType >::value,
            "The mesh entity belongs to other mesh type then the interpolants." );         
         
         const typename MeshEntity::template NeighbourEntities< 1 >& neighbourEntities = entity.getNeighbourEntities();      
                  
         return 0.25 * ( u[ neighbourEntities.template getEntityIndex< -1,  0 >() ] + 
                         u[ neighbourEntities.template getEntityIndex<  1,  0 >() ] +
                         u[ neighbourEntities.template getEntityIndex<  0,  1 >() ] + 
                         u[ neighbourEntities.template getEntityIndex<  0, -1 >() ] );
      }
};

/***
 * 2D grid mesh entity interpolation: 0 -> 2 
 */
template< typename Real,
          typename Device,
          typename Index >
class tnlMeshEntitiesInterpolants< tnlGrid< 2, Real, Device, Index >, 0, 2 >
   : public tnlDomain< 2, MeshInteriorDomain >
{
   public:
      
      typedef tnlGrid< 2, Real, Device, Index > MeshType;

      template< typename MeshFunction, typename MeshEntity >
      __cuda_callable__
      Real operator()( const MeshFunction& u,
                       const MeshEntity& entity,
                       const Real& time = 0.0 ) const
      {
         static_assert( MeshFunction::getEntityDimensions() == 1,
            "Mesh function must be defined on vertices." );

         static_assert( std::is_same< typename MeshEntity::MeshType, MeshType >::value,
            "The mesh entity belongs to other mesh type then the interpolants." );         
         
         const typename MeshEntity::template NeighbourEntities< 0 >& neighbourEntities = entity.getNeighbourEntities();      
                  
         return 0.25 * ( u[ neighbourEntities.template getEntityIndex< -1,  1 >() ] + 
                         u[ neighbourEntities.template getEntityIndex<  1,  1 >() ] +
                         u[ neighbourEntities.template getEntityIndex< -1, -1 >() ] + 
                         u[ neighbourEntities.template getEntityIndex<  1, -1 >() ] );
      }
};

/***
 * 3D grid mesh entity interpolation: 3 -> 2 
 */
template< typename Real,
          typename Device,
          typename Index >
class tnlMeshEntitiesInterpolants< tnlGrid< 3, Real, Device, Index >, 3, 2 >
   : public tnlDomain< 3, MeshInteriorDomain >
{
   public:
      
      typedef tnlGrid< 3, Real, Device, Index > MeshType;

      template< typename MeshFunction, typename MeshEntity >
      __cuda_callable__
      Real operator()( const MeshFunction& u,
                       const MeshEntity& entity,
                       const Real& time = 0.0 ) const
      {
         static_assert( MeshFunction::getEntityDimensions() == 3,
            "Mesh function must be defined on cells." );

         static_assert( std::is_same< typename MeshEntity::MeshType, MeshType >::value,
            "The mesh entity belongs to other mesh type then the interpolants." );         
         
         const typename MeshEntity::template NeighbourEntities< 3 >& neighbourEntities = entity.getNeighbourEntities();      
                  
         if( entity.getOrientation().x() == 1.0 )
            return 0.5 * ( u[ neighbourEntities.template getEntityIndex< -1,  0,  0 >() ] + 
                           u[ neighbourEntities.template getEntityIndex<  1,  0,  0 >() ] );
         if( entity.getOrientation().y() == 1.0 )
            return 0.5 * ( u[ neighbourEntities.template getEntityIndex<  0, -1,  0 >() ] + 
                           u[ neighbourEntities.template getEntityIndex<  0,  1,  0 >() ] );
         else
            return 0.5 * ( u[ neighbourEntities.template getEntityIndex<  0,  0, -1 >() ] + 
                           u[ neighbourEntities.template getEntityIndex<  0,  0,  1 >() ] );            
      }
};

/***
 * 3D grid mesh entity interpolation: 2 -> 3 
 */
template< typename Real,
          typename Device,
          typename Index >
class tnlMeshEntitiesInterpolants< tnlGrid< 3, Real, Device, Index >, 2, 3 >
   : public tnlDomain< 3, MeshInteriorDomain >
{
   public:
      
      typedef tnlGrid< 3, Real, Device, Index > MeshType;

      template< typename MeshFunction, typename MeshEntity >
      __cuda_callable__
      Real operator()( const MeshFunction& u,
                       const MeshEntity& entity,
                       const Real& time = 0.0 ) const
      {
         static_assert( MeshFunction::getEntityDimensions() == 3,
            "Mesh function must be defined on faces." );

         static_assert( std::is_same< typename MeshEntity::MeshType, MeshType >::value,
            "The mesh entity belongs to other mesh type then the interpolants." );         
         
         const typename MeshEntity::template NeighbourEntities< 2 >& neighbourEntities = entity.getNeighbourEntities();      
                  
         return 1.0 / 6.0 * ( u[ neighbourEntities.template getEntityIndex< -1,  0,  0 >() ] + 
                              u[ neighbourEntities.template getEntityIndex<  1,  0,  0 >() ] +
                              u[ neighbourEntities.template getEntityIndex<  0, -1,  0 >() ] + 
                              u[ neighbourEntities.template getEntityIndex<  0,  1,  0 >() ] +
                              u[ neighbourEntities.template getEntityIndex<  0,  0, -1 >() ] + 
                              u[ neighbourEntities.template getEntityIndex<  0,  0,  1 >() ] );            
      }
};

#endif	/* TNLMESHENTITIESINTERPOLANTS_H */

