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
#include <TNL/Solvers/BuildConfigTags.h>
#include <TNL/Meshes/Readers/TNL.h>

namespace TNL {
namespace Solvers {

template< typename ConfigTag,
          typename Device,
          template< typename MeshType > class ProblemSetter >
class MeshTypeResolver
{
   public:

   static bool run( const Config::ParameterContainer& parameters );

   protected:

   static bool resolveMeshDimension( const Config::ParameterContainer& parameters,
                                      Meshes::Readers::TNL& reader );

   // Overload for disabled dimensions
   template< int MeshDimension,
             typename = typename std::enable_if< ! ConfigTagDimensions< ConfigTag,MeshDimension>::enabled >::type,
             typename = void >
   static bool resolveMeshRealType( const Config::ParameterContainer& parameters,
                                    Meshes::Readers::TNL& reader );

   // Overload for enabled dimensions
   template< int MeshDimension,
             typename = typename std::enable_if< ConfigTagDimensions< ConfigTag,MeshDimension>::enabled >::type >
   static bool resolveMeshRealType( const Config::ParameterContainer& parameters,
                                    Meshes::Readers::TNL& reader );

   // Overload for disabled real types
   template< int MeshDimension,
             typename MeshRealType,
             typename = typename std::enable_if< ! ConfigTagReal< ConfigTag, MeshRealType>::enabled >::type,
             typename = void >
   static bool resolveMeshIndexType( const Config::ParameterContainer& parameters,
                                     Meshes::Readers::TNL& reader );

   // Overload for enabled real types
   template< int MeshDimension,
             typename MeshRealType,
             typename = typename std::enable_if< ConfigTagReal< ConfigTag, MeshRealType>::enabled >::type >
   static bool resolveMeshIndexType( const Config::ParameterContainer& parameters,
                                     Meshes::Readers::TNL& reader );

   // Overload for disabled index types
   template< int MeshDimension,
             typename MeshRealType,
             typename MeshIndexType,
             typename = typename std::enable_if< ! ConfigTagIndex< ConfigTag, MeshIndexType >::enabled >::type,
             typename = void >
   static bool resolveMeshType( const Config::ParameterContainer& parameters,
                                Meshes::Readers::TNL& reader );

   // Overload for enabled index types
   template< int MeshDimension,
             typename MeshRealType,
             typename MeshIndexType,
             typename = typename std::enable_if< ConfigTagIndex< ConfigTag, MeshIndexType >::enabled >::type >
   static bool resolveMeshType( const Config::ParameterContainer& parameters,
                                Meshes::Readers::TNL& reader );

   // Overload for disabled mesh types
   template< typename MeshType,
             typename = typename std::enable_if< ! ConfigTagMesh< ConfigTag, MeshType >::enabled >::type,
             typename = void >
   static bool resolveTerminate( const Config::ParameterContainer& parameters,
                                 Meshes::Readers::TNL& reader );

   // Overload for enabled mesh types
   template< typename MeshType,
             typename = typename std::enable_if< ConfigTagMesh< ConfigTag, MeshType >::enabled >::type >
   static bool resolveTerminate( const Config::ParameterContainer& parameters,
                                 Meshes::Readers::TNL& reader );
};

} // namespace Solvers
} // namespace TNL

#include <TNL/Solvers/MeshTypeResolver_impl.h>
