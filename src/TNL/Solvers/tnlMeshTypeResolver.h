/***************************************************************************
                          tnlMeshTypeResolver.h  -  description
                             -------------------
    begin                : Nov 28, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Config/ParameterContainer.h>

namespace TNL {

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

   static bool run( const Config::ParameterContainer& parameters );
};

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename Device,
          typename Index,
          typename ConfigTag  >
class tnlMeshTypeResolver< ProblemSetter, Real, Device, Index, ConfigTag, true >
{
   public:

   static bool run( const Config::ParameterContainer& parameters );

   protected:

   static bool resolveMeshDimensions( const Config::ParameterContainer& parameters,
                                      const List< String >& parsedMeshType );

   // Overload for disabled dimensions
   template< int MeshDimensions,
             typename = typename std::enable_if< ! tnlConfigTagDimensions<ConfigTag,MeshDimensions>::enabled >::type,
             typename = void >
   static bool resolveMeshRealType( const Config::ParameterContainer& parameters,
                                    const List< String >& parsedMeshType );

   // Overload for enabled dimensions
   template< int MeshDimensions,
             typename = typename std::enable_if< tnlConfigTagDimensions<ConfigTag,MeshDimensions>::enabled >::type >
   static bool resolveMeshRealType( const Config::ParameterContainer& parameters,
                                    const List< String >& parsedMeshType );

   // Overload for disabled real types
   template< int MeshDimensions,
             typename MeshRealType,
             typename = typename std::enable_if< ! tnlConfigTagReal<ConfigTag, MeshRealType>::enabled >::type,
             typename = void >
   static bool resolveMeshIndexType( const Config::ParameterContainer& parameters,
                                     const List< String >& parsedMeshType );

   // Overload for enabled real types
   template< int MeshDimensions,
             typename MeshRealType,
             typename = typename std::enable_if< tnlConfigTagReal<ConfigTag, MeshRealType>::enabled >::type >
   static bool resolveMeshIndexType( const Config::ParameterContainer& parameters,
                                     const List< String >& parsedMeshType );

   // Overload for disabled index types
   template< int MeshDimensions,
             typename MeshRealType,
             typename MeshIndexType,
             typename = typename std::enable_if< ! tnlConfigTagIndex<ConfigTag, MeshIndexType>::enabled >::type,
             typename = void >
   static bool resolveMeshType( const Config::ParameterContainer& parameters,
                                const List< String >& parsedMeshType );

   // Overload for enabled index types
   template< int MeshDimensions,
             typename MeshRealType,
             typename MeshIndexType,
             typename = typename std::enable_if< tnlConfigTagIndex<ConfigTag, MeshIndexType>::enabled >::type >
   static bool resolveMeshType( const Config::ParameterContainer& parameters,
                                const List< String >& parsedMeshType );



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

   static bool checkDimensions( const Config::ParameterContainer& parameters,
                                const List< String >& parsedMeshType );
};

template< int Dimensions, typename MeshTypeResolver >
class tnlMeshTypeResolverDimensionsSupportChecker< Dimensions, false, MeshTypeResolver >
{
   public:

   static bool checkDimensions( const Config::ParameterContainer& parameters,
                                const List< String >& parsedMeshType );
};*/

} // namespace TNL

#include <TNL/Solvers/tnlMeshTypeResolver_impl.h>
