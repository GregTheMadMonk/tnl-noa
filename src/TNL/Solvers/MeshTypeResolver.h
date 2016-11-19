/***************************************************************************
                          MeshTypeResolver.h  -  description
                             -------------------
    begin                : Nov 28, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Config/ParameterContainer.h>

namespace TNL {
namespace Solvers {   

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename Device,
          typename Index,
          typename ConfigTag,
          bool ResolveMesh = ConfigTagMeshResolve< ConfigTag >::enabled >
class MeshTypeResolver
{
};

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename Device,
          typename Index,
          typename ConfigTag >
class MeshTypeResolver< ProblemSetter, Real, Device, Index, ConfigTag, false >
{
   public:

   static bool run( const Config::ParameterContainer& parameters );
};

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename Device,
          typename Index,
          typename ConfigTag  >
class MeshTypeResolver< ProblemSetter, Real, Device, Index, ConfigTag, true >
{
   public:

   static bool run( const Config::ParameterContainer& parameters );

   protected:

   static bool resolveMeshDimensions( const Config::ParameterContainer& parameters,
                                      const Containers::List< String >& parsedMeshType );

   // Overload for disabled dimensions
   template< int MeshDimensions,
             typename = typename std::enable_if< ! ConfigTagDimensions<ConfigTag,MeshDimensions>::enabled >::type,
             typename = void >
   static bool resolveMeshRealType( const Config::ParameterContainer& parameters,
                                    const Containers::List< String >& parsedMeshType );

   // Overload for enabled dimensions
   template< int MeshDimensions,
             typename = typename std::enable_if< ConfigTagDimensions<ConfigTag,MeshDimensions>::enabled >::type >
   static bool resolveMeshRealType( const Config::ParameterContainer& parameters,
                                    const Containers::List< String >& parsedMeshType );

   // Overload for disabled real types
   template< int MeshDimensions,
             typename MeshRealType,
             typename = typename std::enable_if< ! ConfigTagReal<ConfigTag, MeshRealType>::enabled >::type,
             typename = void >
   static bool resolveMeshIndexType( const Config::ParameterContainer& parameters,
                                     const Containers::List< String >& parsedMeshType );

   // Overload for enabled real types
   template< int MeshDimensions,
             typename MeshRealType,
             typename = typename std::enable_if< ConfigTagReal<ConfigTag, MeshRealType>::enabled >::type >
   static bool resolveMeshIndexType( const Config::ParameterContainer& parameters,
                                     const Containers::List< String >& parsedMeshType );

   // Overload for disabled index types
   template< int MeshDimensions,
             typename MeshRealType,
             typename MeshIndexType,
             typename = typename std::enable_if< ! ConfigTagIndex<ConfigTag, MeshIndexType>::enabled >::type,
             typename = void >
   static bool resolveMeshType( const Config::ParameterContainer& parameters,
                                const Containers::List< String >& parsedMeshType );

   // Overload for enabled index types
   template< int MeshDimensions,
             typename MeshRealType,
             typename MeshIndexType,
             typename = typename std::enable_if< ConfigTagIndex<ConfigTag, MeshIndexType>::enabled >::type >
   static bool resolveMeshType( const Config::ParameterContainer& parameters,
                                const Containers::List< String >& parsedMeshType );



   template< int Dimensions, bool DimensionsSupport, typename MeshTypeResolver >
    friend class MeshTypeResolverDimensionsSupportChecker;
};

/*template< int Dimensions, bool DimensionsSupport, typename MeshTypeResolver >
class MeshTypeResolverDimensionsSupportChecker
{
};

template< int Dimensions, typename MeshTypeResolver >
class MeshTypeResolverDimensionsSupportChecker< Dimensions, true, MeshTypeResolver >
{
   public:

   static bool checkDimensions( const Config::ParameterContainer& parameters,
                                const Containers::List< String >& parsedMeshType );
};

template< int Dimensions, typename MeshTypeResolver >
class MeshTypeResolverDimensionsSupportChecker< Dimensions, false, MeshTypeResolver >
{
   public:

   static bool checkDimensions( const Config::ParameterContainer& parameters,
                                const Containers::List< String >& parsedMeshType );
};*/

} // namespace Solvers
} // namespace TNL

#include <TNL/Solvers/MeshTypeResolver_impl.h>
