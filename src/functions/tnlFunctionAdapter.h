/***************************************************************************
                          tnlFunctionAdapter.h  -  description
                             -------------------
    begin                : Nov 9, 2015
    copyright            : (C) 2015 by oberhuber
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

#ifndef TNLFUNCTIONADAPTER_H
#define	TNLFUNCTIONADAPTER_H

/***
 * MeshType is a type of mesh on which we evaluate the function.
 * DomainType (defined in functions/tnlDomain.h) defines a domain of
 * the function. In TNL, we mostly work with mesh functions. In this case
 * mesh entity and time is passed to the function...
 */
template< typename Mesh,
          typename Function,
          int domainType = Function::getDomainType() >
class tnlFunctionAdapter
{
      public:
      
      typedef Function FunctionType;
      typedef Mesh MeshType;
      typedef typename FunctionType::RealType  RealType;
      typedef typename MeshType::IndexType     IndexType;      
      //typedef typename FunctionType::VertexType VertexType;
      
      template< typename EntityType >
      __cuda_callable__ inline
      static RealType getValue( const FunctionType& function,
                                const EntityType& meshEntity,
                                const RealType& time )
      {         
         return function( meshEntity, time );
      }
};

/***
 * Specialization for analytic functions. In this case
 * we pass vertex and time to the function ...
 */
template< typename Mesh,
          typename Function >
class tnlFunctionAdapter< Mesh, Function, SpaceDomain >
{
   public:
      
      typedef Function FunctionType;
      typedef Mesh MeshType;
      typedef typename FunctionType::RealType  RealType;
      typedef typename MeshType::IndexType     IndexType;      
      typedef typename FunctionType::VertexType VertexType;
      
      template< typename EntityType >
      __cuda_callable__ inline
      static RealType getValue( const FunctionType& function,
                                const EntityType& meshEntity,
                                const RealType& time )
      {         
         return function( meshEntity.getCenter(), time );
      }
};

/***
 * Specialization for analytic space independent functions.
 * Such function does not depend on any space variable and so
 * we pass only time.
 */
template< typename Mesh,
          typename Function >
class tnlFunctionAdapter< Mesh, Function, NonspaceDomain >
{
   public:
      
      typedef Function FunctionType;
      typedef Mesh MeshType;
      typedef typename FunctionType::RealType  RealType;
      typedef typename MeshType::IndexType     IndexType;      
      typedef typename FunctionType::VertexType VertexType;
      
      template< typename EntityType >
      __cuda_callable__ inline
      static RealType getValue( const FunctionType& function,
                                const EntityType& meshEntity,
                                const RealType& time )
      {         
         return function.getValue( time );
      }
};

#ifdef UNDEF

/***
 * Specialization for mesh functions
 */
template< typename Mesh,
          typename Function >
class tnlFunctionAdapter< Mesh, Function, MeshFunction >
{
   public:
      
      typedef Function FunctionType;
      typedef Mesh MeshType;
      typedef typename FunctionType::RealType  RealType;
      typedef typename MeshType::IndexType     IndexType;      
      
      template< typename EntityType >
      __cuda_callable__ inline
      static RealType getValue( const FunctionType& function,
                                const EntityType& meshEntity,
                                const RealType& time )
      {         
         return function( meshEntity, time );
      }
};

/***
 * Specialization for analytic functions
 */
template< typename Mesh,
          typename Function >
class tnlFunctionAdapter< Mesh, Function, SpaceDomain >
{
   public:
      
      typedef Function FunctionType;
      typedef Mesh MeshType;
      typedef typename FunctionType::RealType  RealType;
      typedef typename MeshType::IndexType     IndexType;      
      typedef typename FunctionType::VertexType VertexType;
      
      template< typename EntityType >
      __cuda_callable__ inline
      static RealType getValue( const FunctionType& function,
                                const EntityType& meshEntity,
                                const RealType& time )
      {         
         return function.getValue( meshEntity.getCenter(), time );
      }
};

/***
 * Specialization for constant analytic functions
 */
template< typename Mesh,
          typename Function >
class tnlFunctionAdapter< Mesh, Function, SpaceDomain >
{
   public:
      
      typedef Function FunctionType;
      typedef Mesh MeshType;
      typedef typename FunctionType::RealType  RealType;
      typedef typename MeshType::IndexType     IndexType;      
      typedef typename FunctionType::VertexType VertexType;
      
      template< typename EntityType >
      __cuda_callable__ inline
      static RealType getValue( const FunctionType& function,
                                const EntityType& meshEntity,
                                const RealType& time )
      {         
         return function.getValue( time );
      }
};
#endif

#endif	/* TNLFUNCTIONADAPTER_H */

