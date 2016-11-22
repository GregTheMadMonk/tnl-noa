/***************************************************************************
                          MeshResolver_impl.h  -  description
                             -------------------
    begin                : Nov 22, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <string>
#include <utility>

#include <TNL/Meshes/TypeResolver/TypeResolver.h>
#include <TNL/Meshes/Readers/TNL.h>
#include <TNL/Meshes/TypeResolver/GridTypeResolver.h>
#include <TNL/Meshes/TypeResolver/MeshTypeResolver.h>

inline bool ends_with( const std::string& value, const std::string& ending )
{
   if (ending.size() > value.size())
      return false;
   return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

namespace TNL {
namespace Meshes {

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
template< typename ConfigTag,
          typename Device,
          template< typename MeshType > class ProblemSetter,
          typename... ProblemSetterArgs >
bool resolveMeshType( const String& fileName_,
                      ProblemSetterArgs&&... problemSetterArgs )
{
   std::string fileName( fileName_.getString() );

   if( ends_with( fileName, ".tnl" ) ) {
      Readers::TNL reader;
      if( ! reader.readFile( fileName_ ) )
         return false;
      if( reader.getMeshType() == "Meshes::Grid" )
         return GridTypeResolver< Readers::TNL, ConfigTag, Device, ProblemSetter, ProblemSetterArgs... >::
            run( reader, std::forward<ProblemSetterArgs>(problemSetterArgs)... );
      else if( reader.getMeshType() == "Meshes::Mesh" )
         return MeshTypeResolver< Readers::TNL, ConfigTag, Device, ProblemSetter, ProblemSetterArgs... >::
            run( reader, std::forward<ProblemSetterArgs>(problemSetterArgs)... );
      else {
         std::cerr << "Unsupported mesh type: " << reader.getMeshType() << std::endl;
         return false;
      }
   }
   // TODO
//   else if( ends_with( fileName, ".vtk" ) ) {
//   }
   else {
      std::cerr << "File '" << fileName << "' has unknown extension. Supported extensions are '.tnl' and '.vtk'." << std::endl;
      return false;
   }
}

} // namespace Meshes
} // namespace TNL
