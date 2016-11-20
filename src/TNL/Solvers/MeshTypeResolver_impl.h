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
#include <TNL/Meshes/DummyMesh.h>
#include <TNL/Solvers/MeshTypeResolver.h>
#include <TNL/Solvers/SolverStarter.h>

namespace TNL {
namespace Solvers {   

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename Device,
          typename Index,
          typename MeshType,
          typename ConfigTag,
          bool MeshTypeSupported = ConfigTagMesh< ConfigTag, MeshType >::enabled >
class MeshResolverTerminator{};


template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename Device,
          typename Index,
          typename ConfigTag >
bool MeshTypeResolver< ProblemSetter, Real, Device, Index, ConfigTag, false >::run( const Config::ParameterContainer& parameters )
{
   return ProblemSetter< Real,
                         Device,
                         Index,
                         Meshes::DummyMesh< Real, Device, Index >,
                         ConfigTag,
                         SolverStarter< ConfigTag > >::template run< Real, Device, Index, ConfigTag >( parameters );
};

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename Device,
          typename Index,
          typename ConfigTag >
bool MeshTypeResolver< ProblemSetter, Real, Device, Index, ConfigTag, true >::run( const Config::ParameterContainer& parameters )
{
   const String& meshFileName = parameters.getParameter< String >( "mesh" );
   Meshes::Readers::TNL reader;
   if( ! reader.readFile( meshFileName ) )
      return false;
   return resolveMeshDimension( parameters, reader );
}

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename Device,
          typename Index,
          typename ConfigTag >
bool
MeshTypeResolver< ProblemSetter, Real, Device, Index, ConfigTag, true >::
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

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename Device,
          typename Index,
          typename ConfigTag >
   template< int MeshDimension, typename, typename >
bool
MeshTypeResolver< ProblemSetter, Real, Device, Index, ConfigTag, true >::
resolveMeshRealType( const Config::ParameterContainer& parameters,
                     Meshes::Readers::TNL& reader )
{
   std::cerr << "Mesh dimension " << MeshDimension << " is not supported." << std::endl;
   return false;
}

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename Device,
          typename Index,
          typename ConfigTag >
   template< int MeshDimension, typename >
bool
MeshTypeResolver< ProblemSetter, Real, Device, Index, ConfigTag, true >::
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

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename Device,
          typename Index,
          typename ConfigTag >
   template< int MeshDimension,
             typename MeshRealType,
             typename, typename >
bool
MeshTypeResolver< ProblemSetter, Real, Device, Index, ConfigTag, true >::
resolveMeshIndexType( const Config::ParameterContainer& parameters,
                      Meshes::Readers::TNL& reader )
{
   std::cerr << "The type '" << reader.getRealType() << "' is not allowed for real type." << std::endl;
   return false;
}

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename Device,
          typename Index,
          typename ConfigTag >
   template< int MeshDimension,
             typename MeshRealType,
             typename >
bool
MeshTypeResolver< ProblemSetter, Real, Device, Index, ConfigTag, true >::
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

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename Device,
          typename Index,
          typename ConfigTag >
   template< int MeshDimension,
             typename MeshRealType,
             typename MeshIndexType,
             typename, typename >
bool
MeshTypeResolver< ProblemSetter, Real, Device, Index, ConfigTag, true >::
resolveMeshType( const Config::ParameterContainer& parameters,
                 Meshes::Readers::TNL& reader )
{
   std::cerr << "The type '" << reader.getIndexType() << "' is not allowed for indexing type." << std::endl;
   return false;
}

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename Device,
          typename Index,
          typename ConfigTag >
   template< int MeshDimension,
             typename MeshRealType,
             typename MeshIndexType,
             typename >
bool
MeshTypeResolver< ProblemSetter, Real, Device, Index, ConfigTag, true >::
resolveMeshType( const Config::ParameterContainer& parameters,
                 Meshes::Readers::TNL& reader )
{
   if( reader.getMeshType() == "Meshes::Grid" )
   {
      using MeshType = Meshes::Grid< MeshDimension, MeshRealType, Device, MeshIndexType >;
      // TODO: the mesh can be loaded here (this will be probably necessary for unstructured meshes anyway)
      return MeshResolverTerminator< ProblemSetter, Real, Device, Index, MeshType, ConfigTag >::run( parameters );
   }
   std::cerr << "Unknown mesh type " << reader.getMeshType() << "." << std::endl;
   return false;
}

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename Device,
          typename Index,
          typename MeshType,
          typename ConfigTag >
class MeshResolverTerminator< ProblemSetter, Real, Device, Index, MeshType, ConfigTag, false >
{
   public:
      static bool run( const Config::ParameterContainer& parameters )
      {
         std::cerr << "The mesh type " << TNL::getType< MeshType >() << " is not supported." << std::endl;
         return false;
      };
};

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename Device,
          typename Index,
          typename MeshType,
          typename ConfigTag >
class MeshResolverTerminator< ProblemSetter, Real, Device, Index, MeshType, ConfigTag, true >
{
   public:
      static bool run( const Config::ParameterContainer& parameters )
      {
         return ProblemSetter< Real, Device, Index, MeshType, ConfigTag, SolverStarter< ConfigTag > >::run( parameters );
      }
};

} // namespace Solvers
} // namespace TNL
