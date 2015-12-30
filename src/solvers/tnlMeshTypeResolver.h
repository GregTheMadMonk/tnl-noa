/***************************************************************************
                          tnlMeshTypeResolver.h  -  description
                             -------------------
    begin                : Nov 28, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
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

#ifndef TNLMESHTYPERESOLVER_H_
#define TNLMESHTYPERESOLVER_H_

#include <config/tnlParameterContainer.h>

template< template< typename Real, typename Device, typename Index, typename MeshType, typename MeshConfig, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename Device,
          typename Index,
          typename MeshConfig,
          bool ResolveMesh = tnlMeshConfigMeshResolve< MeshConfig >::enabled >
class tnlMeshTypeResolver
{
};

template< template< typename Real, typename Device, typename Index, typename MeshType, typename MeshConfig, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename Device,
          typename Index,
          typename MeshConfig >
class tnlMeshTypeResolver< ProblemSetter, Real, Device, Index, MeshConfig, false >
{
   public:

   static bool run( const tnlParameterContainer& parameters );
};

template< template< typename Real, typename Device, typename Index, typename MeshType, typename MeshConfig, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename Device,
          typename Index,
          typename MeshConfig  >
class tnlMeshTypeResolver< ProblemSetter, Real, Device, Index, MeshConfig, true >
{
   public:

   static bool run( const tnlParameterContainer& parameters );

   protected:

   static bool resolveMeshDimensions( const tnlParameterContainer& parameters,
                                      const tnlList< tnlString >& parsedMeshType );

   // Overload for disabled dimensions
   template< int MeshDimensions,
             typename = typename std::enable_if< ! tnlMeshConfigDimensions<MeshConfig,MeshDimensions>::enabled >::type,
             typename = void >
   static bool resolveMeshRealType( const tnlParameterContainer& parameters,
                                    const tnlList< tnlString >& parsedMeshType );

   // Overload for enabled dimensions
   template< int MeshDimensions,
             typename = typename std::enable_if< tnlMeshConfigDimensions<MeshConfig,MeshDimensions>::enabled >::type >
   static bool resolveMeshRealType( const tnlParameterContainer& parameters,
                                    const tnlList< tnlString >& parsedMeshType );

   // Overload for disabled real types
   template< int MeshDimensions,
             typename MeshRealType,
             typename = typename std::enable_if< ! tnlMeshConfigReal<MeshConfig, MeshRealType>::enabled >::type,
             typename = void >
   static bool resolveMeshIndexType( const tnlParameterContainer& parameters,
                                     const tnlList< tnlString >& parsedMeshType );

   // Overload for enabled real types
   template< int MeshDimensions,
             typename MeshRealType,
             typename = typename std::enable_if< tnlMeshConfigReal<MeshConfig, MeshRealType>::enabled >::type >
   static bool resolveMeshIndexType( const tnlParameterContainer& parameters,
                                     const tnlList< tnlString >& parsedMeshType );

   // Overload for disabled index types
   template< int MeshDimensions,
             typename MeshRealType,
             typename MeshIndexType,
             typename = typename std::enable_if< ! tnlMeshConfigIndex<MeshConfig, MeshIndexType>::enabled >::type,
             typename = void >
   static bool resolveMeshType( const tnlParameterContainer& parameters,
                                const tnlList< tnlString >& parsedMeshType );

   // Overload for enabled index types
   template< int MeshDimensions,
             typename MeshRealType,
             typename MeshIndexType,
             typename = typename std::enable_if< tnlMeshConfigIndex<MeshConfig, MeshIndexType>::enabled >::type >
   static bool resolveMeshType( const tnlParameterContainer& parameters,
                                const tnlList< tnlString >& parsedMeshType );



   template< int Dimensions, bool DimensionsSupport, typename MeshTypeResolver >
    friend class tnlMeshTypeResolverDimensionsSupportChecker;
};

/*template< int Dimensions, bool DimensionsSupport, typename MeshTypeResolver >
class tnlMeshTypeResolverDimensionsSupportChecker
{
};

template< int Dimensions, typename MeshTypeResolver >
class tnlMeshTypeResolverDimensionsSupportChecker< Dimensions, true, MeshTypeResolver >
{
   public:

   static bool checkDimensions( const tnlParameterContainer& parameters,
                                const tnlList< tnlString >& parsedMeshType );
};

template< int Dimensions, typename MeshTypeResolver >
class tnlMeshTypeResolverDimensionsSupportChecker< Dimensions, false, MeshTypeResolver >
{
   public:

   static bool checkDimensions( const tnlParameterContainer& parameters,
                                const tnlList< tnlString >& parsedMeshType );
};*/

#include <solvers/tnlMeshTypeResolver_impl.h>

#endif /* TNLMESHTYPERESOLVER_H_ */
