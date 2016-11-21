/***************************************************************************
                          MeshTypeResolver.h  -  description
                             -------------------
    begin                : Nov 28, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Solvers/BuildConfigTags.h>
#include <TNL/Meshes/Readers/TNL.h>

/*
 * TODO:
 * The variadic template parameter pack ProblemSetterArgs will not be necessary
 * in C++14 as it will be possible to use generic lambda functions to pass
 * parameters to the ProblemSetter:
 *
 *    // wrapper for MeshTypeResolver
 *    template< typename MeshType >
 *    using ProblemSetterWrapper = ProblemSetter< Real, Device, Index, MeshType, ConfigTag, SolverStarter< ConfigTag > >;
 *
 *    bool run( const Config::ParameterContainer& parameters )
 *    {
 *       const String& meshFileName = parameters.getParameter< String >( "mesh" );
 *       auto wrapper = []( auto&& mesh ) {
 *           return ProblemSetterWrapper< decltype(mesh) >::run( parameters );
 *       };
 *       return MeshTypeResolver< ConfigTag, Device, wrapper >::run( meshFileName );
 *    }
 */

namespace TNL {
namespace Solvers {

template< typename ConfigTag,
          typename Device,
          template< typename MeshType > class ProblemSetter,
          typename... ProblemSetterArgs >
class MeshTypeResolver
{
   public:

   static bool run( const String& fileName,
                    ProblemSetterArgs&&... problemSetterArgs );

   protected:

   static bool resolveMeshDimension( Meshes::Readers::TNL& reader,
                                     ProblemSetterArgs&&... problemSetterArgs );

   // Overload for disabled dimensions
   template< int MeshDimension,
             typename = typename std::enable_if< ! ConfigTagDimensions< ConfigTag,MeshDimension>::enabled >::type,
             typename = void >
   static bool resolveMeshRealType( Meshes::Readers::TNL& reader,
                                    ProblemSetterArgs&&... problemSetterArgs );

   // Overload for enabled dimensions
   template< int MeshDimension,
             typename = typename std::enable_if< ConfigTagDimensions< ConfigTag,MeshDimension>::enabled >::type >
   static bool resolveMeshRealType( Meshes::Readers::TNL& reader,
                                    ProblemSetterArgs&&... problemSetterArgs );

   // Overload for disabled real types
   template< int MeshDimension,
             typename MeshRealType,
             typename = typename std::enable_if< ! ConfigTagReal< ConfigTag, MeshRealType>::enabled >::type,
             typename = void >
   static bool resolveMeshIndexType( Meshes::Readers::TNL& reader,
                                     ProblemSetterArgs&&... problemSetterArgs );

   // Overload for enabled real types
   template< int MeshDimension,
             typename MeshRealType,
             typename = typename std::enable_if< ConfigTagReal< ConfigTag, MeshRealType>::enabled >::type >
   static bool resolveMeshIndexType( Meshes::Readers::TNL& reader,
                                     ProblemSetterArgs&&... problemSetterArgs );

   // Overload for disabled index types
   template< int MeshDimension,
             typename MeshRealType,
             typename MeshIndexType,
             typename = typename std::enable_if< ! ConfigTagIndex< ConfigTag, MeshIndexType >::enabled >::type,
             typename = void >
   static bool resolveMeshType( Meshes::Readers::TNL& reader,
                                ProblemSetterArgs&&... problemSetterArgs );

   // Overload for enabled index types
   template< int MeshDimension,
             typename MeshRealType,
             typename MeshIndexType,
             typename = typename std::enable_if< ConfigTagIndex< ConfigTag, MeshIndexType >::enabled >::type >
   static bool resolveMeshType( Meshes::Readers::TNL& reader,
                                ProblemSetterArgs&&... problemSetterArgs );

   // Overload for disabled mesh types
   template< typename MeshType,
             typename = typename std::enable_if< ! ConfigTagMesh< ConfigTag, MeshType >::enabled >::type,
             typename = void >
   static bool resolveTerminate( Meshes::Readers::TNL& reader,
                                 ProblemSetterArgs&&... problemSetterArgs );

   // Overload for enabled mesh types
   template< typename MeshType,
             typename = typename std::enable_if< ConfigTagMesh< ConfigTag, MeshType >::enabled >::type >
   static bool resolveTerminate( Meshes::Readers::TNL& reader,
                                 ProblemSetterArgs&&... problemSetterArgs );
};

} // namespace Solvers
} // namespace TNL

#include <TNL/Solvers/MeshTypeResolver_impl.h>
