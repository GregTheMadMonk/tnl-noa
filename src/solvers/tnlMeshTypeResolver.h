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

template< bool ResolveMesh,
          typename Real,
          typename Device,
          typename Index,
          template< typename MeshType, typename SolverStarter > class ProblemSetter >
class tnlMeshTypeResolver
{
};

template< typename Real,
          typename Device,
          typename Index,
          template< typename MeshType, typename SolverStarter > class ProblemSetter >
class tnlMeshTypeResolver< false, Real, Device, Index, ProblemSetter >
{
   public:

   static bool run( const tnlParameterContainer& parameters );
};

template< typename Real,
          typename Device,
          typename Index,
          template< typename MeshType, typename SolverStarter > class ProblemSetter >
class tnlMeshTypeResolver< true, Real, Device, Index, ProblemSetter >
{
   public:

   static bool run( const tnlParameterContainer& parameters );

   protected:

   static bool resolveMeshDimensions( const tnlParameterContainer& parameters,
                                      const tnlList< tnlString >& parsedMeshType );

   template< int MeshDimensions >
   static bool resolveMeshRealType( const tnlParameterContainer& parameters,
                                    const tnlList< tnlString >& parsedMeshType );

   template< int MeshDimensions,
             typename MeshRealType >
   static bool resolveMeshIndexType( const tnlParameterContainer& parameters,
                                     const tnlList< tnlString >& parsedMeshType );

   template< int MeshDimensions,
             typename MeshRealType,
             typename MeshIndexType >
   static bool resolveMeshType( const tnlParameterContainer& parameters,
                                const tnlList< tnlString >& parsedMeshType );

   template< int MeshDimensions,
             typename MeshRealType,
             typename MeshIndexType >
   static bool resolveGridGeometryType( const tnlParameterContainer& parameters,
                                        const tnlList< tnlString >& parsedMeshType );

   template< int Dimensions, bool DimensionsSupport, typename MeshTypeResolver >
    friend class tnlMeshTypeResolverDimensionsSupportChecker;
};

template< int Dimensions, bool DimensionsSupport, typename MeshTypeResolver >
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
};



#include <implementation/solvers/tnlMeshTypeResolver_impl.h>

#endif /* TNLMESHTYPERESOLVER_H_ */