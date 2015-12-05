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
 *  MeshType is a type of mesh on which we evaluate the function
 *  FunctionType is a type of function which we want to evaluate
 *  FunctionMeshType is a type mesh on which the function is "defined"
 *   - it can be void for analytic function given by a formula or
 *     MeshType for mesh functions
 *   - this adapter passes a vertex as a space variable to analytic functions
 *     and mesh entity index for mesh functions.
 */
template< typename Mesh,
          typename Function,
          //tnlFunctionType functionType = Function::functionType >
          int functionType = Function::functionType >
class tnlFunctionAdapter
{
};

/***
 * Specialization for analytic functions
 */
template< typename Mesh,
          typename Function >
class tnlFunctionAdapter< Mesh, Function, tnlAnalyticFunction >
{
   public:
      
      typedef Function FunctionType;
      typedef Mesh MeshType;
      typedef typename FunctionType::RealType  RealType;
      typedef typename MeshType::IndexType     IndexType;      
      typedef typename FunctionType::VertexType VertexType;
      
      template< int EntityDimensions >
      __cuda_callable__ inline
      static RealType getValue( const MeshType& mesh,
                                const FunctionType& function,
                                const IndexType meshEntityIndex,
                                const RealType& time )
      {         
         return function.getValue( mesh.template getEntityCenter< EntityDimensions >( meshEntityIndex ), time );
      };
};

/***
 * Specialization for analytic functions and grids.
 */
template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename Function >
class tnlFunctionAdapter< tnlGrid< Dimensions, Real, Device, Index >, Function, tnlAnalyticFunction >
{
   public:
     
      typedef Function FunctionType; 
      typedef tnlGrid< Dimensions, Real, Device, Index > MeshType;
      typedef typename FunctionType::RealType RealType;
      typedef typename MeshType::IndexType IndexType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      
      template< typename EntityType >
      __cuda_callable__ inline
      static RealType getValue( const MeshType& mesh,
                                const FunctionType& function,
                                const IndexType meshEntytiIndex,
                                const EntityType& entity,
                                const RealType& time )
      {
         return function.getValue( entity.getCenter(), time );
         //return 0.0;
      }
};

/***
 * Specialization for constant functions and grids.
 */
template< int Dimensions,
          typename Real,
          typename Device,
          typename Index,
          typename Function >
class tnlFunctionAdapter< tnlGrid< Dimensions, Real, Device, Index >, Function, tnlAnalyticConstantFunction >
{
   public:
     
      typedef Function FunctionType; 
      typedef tnlGrid< Dimensions, Real, Device, Index > MeshType;
      typedef typename FunctionType::RealType RealType;
      typedef typename MeshType::IndexType IndexType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      
      template< typename EntityType >
      __cuda_callable__ inline
      static RealType getValue( const MeshType& mesh,
                                const FunctionType& function,
                                const IndexType meshEntytiIndex,
                                const EntityType& entity,
                                const RealType& time )
      {
         return function.getValue( time );
      }
};


#endif	/* TNLFUNCTIONADAPTER_H */

