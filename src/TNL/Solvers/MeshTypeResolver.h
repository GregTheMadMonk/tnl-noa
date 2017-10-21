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

   static bool resolveMeshDimension( const Config::ParameterContainer& parameters,
                                      const Containers::List< String >& parsedMeshType );

   // Overload for disabled dimensions
   template< int MeshDimension,
             typename = typename std::enable_if< ! ConfigTagDimension<ConfigTag,MeshDimension>::enabled >::type,
             typename = void >
   static bool resolveMeshRealType( const Config::ParameterContainer& parameters,
                                    const Containers::List< String >& parsedMeshType );

   // Overload for enabled dimensions
   template< int MeshDimension,
             typename = typename std::enable_if< ConfigTagDimension<ConfigTag,MeshDimension>::enabled >::type >
   static bool resolveMeshRealType( const Config::ParameterContainer& parameters,
                                    const Containers::List< String >& parsedMeshType );

   // Overload for disabled real types
   template< int MeshDimension,
             typename MeshRealType,
             typename = typename std::enable_if< ! ConfigTagReal<ConfigTag, MeshRealType>::enabled >::type,
             typename = void >
   static bool resolveMeshIndexType( const Config::ParameterContainer& parameters,
                                     const Containers::List< String >& parsedMeshType );

   // Overload for enabled real types
   template< int MeshDimension,
             typename MeshRealType,
             typename = typename std::enable_if< ConfigTagReal<ConfigTag, MeshRealType>::enabled >::type >
   static bool resolveMeshIndexType( const Config::ParameterContainer& parameters,
                                     const Containers::List< String >& parsedMeshType );

   // Overload for disabled index types
   template< int MeshDimension,
             typename MeshRealType,
             typename MeshIndexType,
             typename = typename std::enable_if< ! ConfigTagIndex<ConfigTag, MeshIndexType>::enabled >::type,
             typename = void >
   static bool resolveMeshType( const Config::ParameterContainer& parameters,
                                const Containers::List< String >& parsedMeshType );

   // Overload for enabled index types
   template< int MeshDimension,
             typename MeshRealType,
             typename MeshIndexType,
             typename = typename std::enable_if< ConfigTagIndex<ConfigTag, MeshIndexType>::enabled >::type >
   static bool resolveMeshType( const Config::ParameterContainer& parameters,
                                const Containers::List< String >& parsedMeshType );



   template< int Dimension, bool DimensionSupport, typename MeshTypeResolver >
    friend class MeshTypeResolverDimensionSupportChecker;
};

/*template< int Dimension, bool DimensionSupport, typename MeshTypeResolver >
class MeshTypeResolverDimensionSupportChecker
{
};

template< int Dimension, typename MeshTypeResolver >
class MeshTypeResolverDimensionSupportChecker< Dimension, true, MeshTypeResolver >
{
   public:

   static bool checkDimension( const Config::ParameterContainer& parameters,
                                const Containers::List< String >& parsedMeshType );
};

template< int Dimension, typename MeshTypeResolver >
class MeshTypeResolverDimensionSupportChecker< Dimension, false, MeshTypeResolver >
{
   public:

   static bool checkDimension( const Config::ParameterContainer& parameters,
                                const Containers::List< String >& parsedMeshType );
};*/

} // namespace Solvers
} // namespace TNL

#include <TNL/Solvers/MeshTypeResolver_impl.h>
