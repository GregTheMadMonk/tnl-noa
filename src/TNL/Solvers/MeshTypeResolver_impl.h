/***************************************************************************
                          MeshTypeResolver_impl.h  -  description
                             -------------------
    begin                : Nov 28, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/String.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/Solvers/MeshTypeResolver.h>

namespace TNL {
namespace Solvers {   

template< typename ConfigTag,
          typename Device,
          template< typename MeshType > class ProblemSetter >
bool
MeshTypeResolver< ConfigTag, Device, ProblemSetter >::
run( const Config::ParameterContainer& parameters )
{
   const String& meshFileName = parameters.getParameter< String >( "mesh" );
   Meshes::Readers::TNL reader;
   if( ! reader.readFile( meshFileName ) )
      return false;
   return resolveMeshDimension( parameters, reader );
}

template< typename ConfigTag,
          typename Device,
          template< typename MeshType > class ProblemSetter >
bool
MeshTypeResolver< ConfigTag, Device, ProblemSetter >::
resolveMeshDimension( const Config::ParameterContainer& parameters,
                       Meshes::Readers::TNL& reader )
{
   if( reader.getMeshDimension() == 1 )
      return resolveMeshRealType< 1 >( parameters, reader );
   if( reader.getMeshDimension() == 2 )
      return resolveMeshRealType< 2 >( parameters, reader );
   if( reader.getMeshDimension() == 3 )
      return resolveMeshRealType< 3 >( parameters, reader );
   std::cerr << "Unsupported mesh dimension: " << reader.getMeshDimension() << std::endl;
   return false;
}

template< typename ConfigTag,
          typename Device,
          template< typename MeshType > class ProblemSetter >
   template< int MeshDimension, typename, typename >
bool
MeshTypeResolver< ConfigTag, Device, ProblemSetter >::
resolveMeshRealType( const Config::ParameterContainer& parameters,
                     Meshes::Readers::TNL& reader )
{
   std::cerr << "Mesh dimension " << MeshDimension << " is not supported." << std::endl;
   return false;
}

template< typename ConfigTag,
          typename Device,
          template< typename MeshType > class ProblemSetter >
   template< int MeshDimension, typename >
bool
MeshTypeResolver< ConfigTag, Device, ProblemSetter >::
resolveMeshRealType( const Config::ParameterContainer& parameters,
                     Meshes::Readers::TNL& reader )
{
   if( reader.getRealType() == "float" )
      return resolveMeshIndexType< MeshDimension, float >( parameters, reader );
   if( reader.getRealType() == "double" )
      return resolveMeshIndexType< MeshDimension, double >( parameters, reader );
   if( reader.getRealType() == "long-double" )
      return resolveMeshIndexType< MeshDimension, long double >( parameters, reader );
   std::cerr << "The type '" << reader.getRealType() << "' is not allowed for real type." << std::endl;
   return false;
}

template< typename ConfigTag,
          typename Device,
          template< typename MeshType > class ProblemSetter >
   template< int MeshDimension,
             typename MeshRealType,
             typename, typename >
bool
MeshTypeResolver< ConfigTag, Device, ProblemSetter >::
resolveMeshIndexType( const Config::ParameterContainer& parameters,
                      Meshes::Readers::TNL& reader )
{
   std::cerr << "The type '" << reader.getRealType() << "' is not allowed for real type." << std::endl;
   return false;
}

template< typename ConfigTag,
          typename Device,
          template< typename MeshType > class ProblemSetter >
   template< int MeshDimension,
             typename MeshRealType,
             typename >
bool
MeshTypeResolver< ConfigTag, Device, ProblemSetter >::
resolveMeshIndexType( const Config::ParameterContainer& parameters,
                      Meshes::Readers::TNL& reader )
{
   if( reader.getIndexType() == "short int" )
      return resolveMeshType< MeshDimension, MeshRealType, short int >( parameters, reader );
   if( reader.getIndexType() == "int" )
      return resolveMeshType< MeshDimension, MeshRealType, int >( parameters, reader );
   if( reader.getIndexType() == "long int" )
      return resolveMeshType< MeshDimension, MeshRealType, long int >( parameters, reader );
   std::cerr << "The type '" << reader.getIndexType() << "' is not allowed for indexing type." << std::endl;
   return false;
}

template< typename ConfigTag,
          typename Device,
          template< typename MeshType > class ProblemSetter >
   template< int MeshDimension,
             typename MeshRealType,
             typename MeshIndexType,
             typename, typename >
bool
MeshTypeResolver< ConfigTag, Device, ProblemSetter >::
resolveMeshType( const Config::ParameterContainer& parameters,
                 Meshes::Readers::TNL& reader )
{
   std::cerr << "The type '" << reader.getIndexType() << "' is not allowed for indexing type." << std::endl;
   return false;
}

template< typename ConfigTag,
          typename Device,
          template< typename MeshType > class ProblemSetter >
   template< int MeshDimension,
             typename MeshRealType,
             typename MeshIndexType,
             typename >
bool
MeshTypeResolver< ConfigTag, Device, ProblemSetter >::
resolveMeshType( const Config::ParameterContainer& parameters,
                 Meshes::Readers::TNL& reader )
{
   if( reader.getMeshType() == "Meshes::Grid" )
   {
      using MeshType = Meshes::Grid< MeshDimension, MeshRealType, Device, MeshIndexType >;
      return resolveTerminate< MeshType >( parameters, reader );
   }
   std::cerr << "Unknown mesh type " << reader.getMeshType() << "." << std::endl;
   return false;
}

template< typename ConfigTag,
          typename Device,
          template< typename MeshType > class ProblemSetter >
   template< typename MeshType,
             typename, typename >
bool
MeshTypeResolver< ConfigTag, Device, ProblemSetter >::
resolveTerminate( const Config::ParameterContainer& parameters,
                  Meshes::Readers::TNL& reader )
{
   std::cerr << "The mesh type " << TNL::getType< MeshType >() << " is not supported." << std::endl;
   return false;
};

template< typename ConfigTag,
          typename Device,
          template< typename MeshType > class ProblemSetter >
   template< typename MeshType,
             typename >
bool
MeshTypeResolver< ConfigTag, Device, ProblemSetter >::
resolveTerminate( const Config::ParameterContainer& parameters,
                  Meshes::Readers::TNL& reader )
{
   // TODO: the mesh can be loaded here (this will be probably necessary for unstructured meshes anyway)
   return ProblemSetter< MeshType >::run( parameters );
};

} // namespace Solvers
} // namespace TNL
