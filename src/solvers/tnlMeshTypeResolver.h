/***************************************************************************
                          tnlMeshTypeResolver.h  -  description
                             -------------------
    begin                : Nov 28, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLMESHTYPERESOLVER_H_
#define TNLMESHTYPERESOLVER_H_

#include <config/tnlParameterContainer.h>

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename Device,
          typename Index,
          typename ConfigTag,
          bool ResolveMesh = tnlConfigTagMeshResolve< ConfigTag >::enabled >
class tnlMeshTypeResolver
{
};

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename Device,
          typename Index,
          typename ConfigTag >
class tnlMeshTypeResolver< ProblemSetter, Real, Device, Index, ConfigTag, false >
{
   public:

   static bool run( const tnlParameterContainer& parameters );
};

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename Device,
          typename Index,
          typename ConfigTag  >
class tnlMeshTypeResolver< ProblemSetter, Real, Device, Index, ConfigTag, true >
{
   public:

   static bool run( const tnlParameterContainer& parameters );

   protected:

   static bool resolveMeshDimensions( const tnlParameterContainer& parameters,
                                      const tnlList< tnlString >& parsedMeshType );

   // Overload for disabled dimensions
   template< int MeshDimensions,
             typename = typename std::enable_if< ! tnlConfigTagDimensions<ConfigTag,MeshDimensions>::enabled >::type,
             typename = void >
   static bool resolveMeshRealType( const tnlParameterContainer& parameters,
                                    const tnlList< tnlString >& parsedMeshType );

   // Overload for enabled dimensions
   template< int MeshDimensions,
             typename = typename std::enable_if< tnlConfigTagDimensions<ConfigTag,MeshDimensions>::enabled >::type >
   static bool resolveMeshRealType( const tnlParameterContainer& parameters,
                                    const tnlList< tnlString >& parsedMeshType );

   // Overload for disabled real types
   template< int MeshDimensions,
             typename MeshRealType,
             typename = typename std::enable_if< ! tnlConfigTagReal<ConfigTag, MeshRealType>::enabled >::type,
             typename = void >
   static bool resolveMeshIndexType( const tnlParameterContainer& parameters,
                                     const tnlList< tnlString >& parsedMeshType );

   // Overload for enabled real types
   template< int MeshDimensions,
             typename MeshRealType,
             typename = typename std::enable_if< tnlConfigTagReal<ConfigTag, MeshRealType>::enabled >::type >
   static bool resolveMeshIndexType( const tnlParameterContainer& parameters,
                                     const tnlList< tnlString >& parsedMeshType );

   // Overload for disabled index types
   template< int MeshDimensions,
             typename MeshRealType,
             typename MeshIndexType,
             typename = typename std::enable_if< ! tnlConfigTagIndex<ConfigTag, MeshIndexType>::enabled >::type,
             typename = void >
   static bool resolveMeshType( const tnlParameterContainer& parameters,
                                const tnlList< tnlString >& parsedMeshType );

   // Overload for enabled index types
   template< int MeshDimensions,
             typename MeshRealType,
             typename MeshIndexType,
             typename = typename std::enable_if< tnlConfigTagIndex<ConfigTag, MeshIndexType>::enabled >::type >
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
