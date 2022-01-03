#pragma once

#include <limits>
#include <ostream>

#include <TNL/Endianness.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/Meshes/VTKTraits.h>

namespace TNL {
namespace Meshes {
namespace Writers {
namespace detail {

// legacy VTK files do not support 64-bit integers, even in the BINARY format
inline void
writeInt( VTK::FileFormat format, std::ostream& str, std::int32_t value )
{
   if( format == VTK::FileFormat::binary ) {
      value = forceBigEndian( value );
      str.write( reinterpret_cast<const char*>(&value), sizeof(std::int32_t) );
   }
   else {
      str << value << " ";
   }
}

template< typename Real >
void
writeReal( VTK::FileFormat format, std::ostream& str, Real value )
{
   if( format == VTK::FileFormat::binary ) {
      value = forceBigEndian( value );
      str.write( reinterpret_cast<const char*>(&value), sizeof(Real) );
   }
   else {
      str.precision( std::numeric_limits< Real >::digits10 );
      str << value << " ";
   }
}


// TODO: specialization for disabled entities
// Unstructured meshes, entities
template< typename Mesh, int EntityDimension >
struct VTKMeshEntitiesWriter
{
   static void exec( const Mesh& mesh, std::ostream& str, VTK::FileFormat format )
   {
      using EntityType = typename Mesh::template EntityType< EntityDimension >;
      using Index = typename Mesh::GlobalIndexType;

      const Index entitiesCount = mesh.template getEntitiesCount< EntityType >();
      for( Index i = 0; i < entitiesCount; i++ ) {
         const auto& entity = mesh.template getEntity< EntityType >( i );
         const int verticesPerEntity = entity.template getSubentitiesCount< 0 >();
         writeInt( format, str, verticesPerEntity );
         for( int j = 0; j < verticesPerEntity; j++ )
            writeInt( format, str, entity.template getSubentityIndex< 0 >( j ) );
         if( format == VTK::FileFormat::ascii )
            str << "\n";
      }
   }
};

// Unstructured meshes, vertices
template< typename Mesh >
struct VTKMeshEntitiesWriter< Mesh, 0 >
{
   static void exec( const Mesh& mesh, std::ostream& str, VTK::FileFormat format )
   {
      using EntityType = typename Mesh::template EntityType< 0 >;
      using Index = typename Mesh::GlobalIndexType;

      const Index entitiesCount = mesh.template getEntitiesCount< EntityType >();
      const int verticesPerEntity = 1;
      for( Index i = 0; i < entitiesCount; i++ )
      {
         writeInt( format, str, verticesPerEntity );
         writeInt( format, str, i );
         if( format == VTK::FileFormat::ascii )
            str << "\n";
      }
   }
};

// 1D grids, cells
template< typename MeshReal,
          typename Device,
          typename MeshIndex >
struct VTKMeshEntitiesWriter< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, 1 >
{
   using MeshType = Meshes::Grid< 1, MeshReal, Device, MeshIndex >;

   static void exec( const MeshType& mesh, std::ostream& str, VTK::FileFormat format )
   {
      for( MeshIndex i = 0; i < mesh.getDimensions().x(); i++ )
      {
         writeInt( format, str, 2 );
         writeInt( format, str, i );
         writeInt( format, str, i+1 );
         if( format == VTK::FileFormat::ascii )
            str << "\n";
      }
   }
};

// 1D grids, vertices
template< typename MeshReal,
          typename Device,
          typename MeshIndex >
struct VTKMeshEntitiesWriter< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, 0 >
{
   using MeshType = Meshes::Grid< 1, MeshReal, Device, MeshIndex >;

   static void exec( const MeshType& mesh, std::ostream& str, VTK::FileFormat format )
   {
      for( MeshIndex i = 0; i < mesh.getDimensions().x() + 1; i++ )
      {
         writeInt( format, str, 1 );
         writeInt( format, str, i );
         if( format == VTK::FileFormat::ascii )
            str << "\n";
      }
   }
};

// 2D grids, cells
template< typename MeshReal,
          typename Device,
          typename MeshIndex >
struct VTKMeshEntitiesWriter< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, 2 >
{
   using MeshType = Meshes::Grid< 2, MeshReal, Device, MeshIndex >;

   static void exec( const MeshType& mesh, std::ostream& str, VTK::FileFormat format )
   {
      for( MeshIndex j = 0; j < mesh.getDimensions().y(); j++ )
      for( MeshIndex i = 0; i < mesh.getDimensions().x(); i++ )
      {
         writeInt( format, str, 4 );
         writeInt( format, str, j * ( mesh.getDimensions().x() + 1 ) + i );
         writeInt( format, str, j * ( mesh.getDimensions().x() + 1 ) + i + 1 );
         writeInt( format, str, (j+1) * ( mesh.getDimensions().x() + 1 ) + i );
         writeInt( format, str, (j+1) * ( mesh.getDimensions().x() + 1 ) + i + 1 );
         if( format == VTK::FileFormat::ascii )
            str << "\n";
      }
   }
};

// 2D grids, faces
template< typename MeshReal,
          typename Device,
          typename MeshIndex >
struct VTKMeshEntitiesWriter< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, 1 >
{
   using MeshType = Meshes::Grid< 2, MeshReal, Device, MeshIndex >;

   static void exec( const MeshType& mesh, std::ostream& str, VTK::FileFormat format )
   {
      for( MeshIndex j = 0; j < mesh.getDimensions().y(); j++ )
      for( MeshIndex i = 0; i < ( mesh.getDimensions().x() + 1 ); i++ )
      {
         writeInt( format, str, 2 );
         writeInt( format, str, j * ( mesh.getDimensions().x() + 1 ) + i );
         writeInt( format, str, (j+1) * ( mesh.getDimensions().x() + 1 ) + i );
         if( format == VTK::FileFormat::ascii )
            str << "\n";
      }

      for( MeshIndex j = 0; j < (mesh.getDimensions().y()+1); j++ )
      for( MeshIndex i = 0; i < mesh.getDimensions().x(); i++ )
      {
         writeInt( format, str, 2 );
         writeInt( format, str, j * ( mesh.getDimensions().x() + 1 ) + i );
         writeInt( format, str, j * ( mesh.getDimensions().x() + 1 ) + i + 1 );
         if( format == VTK::FileFormat::ascii )
            str << "\n";
      }
   }
};

// 2D grids, vertices
template< typename MeshReal,
          typename Device,
          typename MeshIndex >
struct VTKMeshEntitiesWriter< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, 0 >
{
   using MeshType = Meshes::Grid< 2, MeshReal, Device, MeshIndex >;

   static void exec( const MeshType& mesh, std::ostream& str, VTK::FileFormat format )
   {
      for( MeshIndex j = 0; j < ( mesh.getDimensions().y() + 1 ); j++ )
      for( MeshIndex i = 0; i < ( mesh.getDimensions().x() + 1 ); i++ )
      {
         writeInt( format, str, 1 );
         writeInt( format, str, j * mesh.getDimensions().x() + i );
         if( format == VTK::FileFormat::ascii )
            str << "\n";
      }
   }
};

// 3D grids, cells
template< typename MeshReal,
          typename Device,
          typename MeshIndex >
struct VTKMeshEntitiesWriter< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, 3 >
{
   using MeshType = Meshes::Grid< 3, MeshReal, Device, MeshIndex >;

   static void exec( const MeshType& mesh, std::ostream& str, VTK::FileFormat format )
   {
      for( MeshIndex k = 0; k < mesh.getDimensions().z(); k++ )
      for( MeshIndex j = 0; j < mesh.getDimensions().y(); j++ )
      for( MeshIndex i = 0; i < mesh.getDimensions().x(); i++ )
      {
         writeInt( format, str, 8 );
         writeInt( format, str, k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i );
         writeInt( format, str, k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i + 1 );
         writeInt( format, str, k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + (j+1) * ( mesh.getDimensions().x() + 1 ) + i );
         writeInt( format, str, k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + (j+1) * ( mesh.getDimensions().x() + 1 ) + i + 1 );
         writeInt( format, str, (k+1) * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i );
         writeInt( format, str, (k+1) * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i + 1 );
         writeInt( format, str, (k+1) * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + (j+1) * ( mesh.getDimensions().x() + 1 ) + i );
         writeInt( format, str, (k+1) * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + (j+1) * ( mesh.getDimensions().x() + 1 ) + i + 1 );
         if( format == VTK::FileFormat::ascii )
            str << "\n";
      }
   }
};

// 3D grids, faces
template< typename MeshReal,
          typename Device,
          typename MeshIndex >
struct VTKMeshEntitiesWriter< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, 2 >
{
   using MeshType = Meshes::Grid< 3, MeshReal, Device, MeshIndex >;

   static void exec( const MeshType& mesh, std::ostream& str, VTK::FileFormat format )
   {
      for( MeshIndex k = 0; k < mesh.getDimensions().z(); k++ )
      for( MeshIndex j = 0; j < mesh.getDimensions().y(); j++ )
      for( MeshIndex i = 0; i <= mesh.getDimensions().x(); i++ )
      {
         writeInt( format, str, 4 );
         writeInt( format, str, k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i );
         writeInt( format, str, k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + (j+1) * ( mesh.getDimensions().x() + 1 ) + i );
         writeInt( format, str, (k+1) * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i );
         writeInt( format, str, (k+1) * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + (j+1) * ( mesh.getDimensions().x() + 1 ) + i );
         if( format == VTK::FileFormat::ascii )
            str << "\n";
      }

      for( MeshIndex k = 0; k < mesh.getDimensions().z(); k++ )
      for( MeshIndex j = 0; j <= mesh.getDimensions().y(); j++ )
      for( MeshIndex i = 0; i < mesh.getDimensions().x(); i++ )
      {
         writeInt( format, str, 4 );
         writeInt( format, str, k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i );
         writeInt( format, str, k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i + 1 );
         writeInt( format, str, (k+1) * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i );
         writeInt( format, str, (k+1) * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i + 1 );
         if( format == VTK::FileFormat::ascii )
            str << "\n";
      }

      for( MeshIndex k = 0; k <= mesh.getDimensions().z(); k++ )
      for( MeshIndex j = 0; j < mesh.getDimensions().y(); j++ )
      for( MeshIndex i = 0; i < mesh.getDimensions().x(); i++ )
      {
         writeInt( format, str, 4 );
         writeInt( format, str, k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i );
         writeInt( format, str, k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i + 1 );
         writeInt( format, str, k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + (j+1) * ( mesh.getDimensions().x() + 1 ) + i );
         writeInt( format, str, k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + (j+1) * ( mesh.getDimensions().x() + 1 ) + i + 1 );
         if( format == VTK::FileFormat::ascii )
            str << "\n";
      }
   }
};

// 3D grids, edges
template< typename MeshReal,
          typename Device,
          typename MeshIndex >
struct VTKMeshEntitiesWriter< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, 1 >
{
   using MeshType = Meshes::Grid< 3, MeshReal, Device, MeshIndex >;

   static void exec( const MeshType& mesh, std::ostream& str, VTK::FileFormat format )
   {
      for( MeshIndex k = 0; k <= mesh.getDimensions().z(); k++ )
      for( MeshIndex j = 0; j <= mesh.getDimensions().y(); j++ )
      for( MeshIndex i = 0; i < mesh.getDimensions().x(); i++ )
      {
         writeInt( format, str, 2 );
         writeInt( format, str, k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i );
         writeInt( format, str, k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i + 1 );
         if( format == VTK::FileFormat::ascii )
            str << "\n";
      }

      for( MeshIndex k = 0; k <= mesh.getDimensions().z(); k++ )
      for( MeshIndex j = 0; j < mesh.getDimensions().y(); j++ )
      for( MeshIndex i = 0; i <= mesh.getDimensions().x(); i++ )
      {
         writeInt( format, str, 2 );
         writeInt( format, str, k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i );
         writeInt( format, str, k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + (j+1) * ( mesh.getDimensions().x() + 1 ) + i );
         if( format == VTK::FileFormat::ascii )
            str << "\n";
      }

      for( MeshIndex k = 0; k < mesh.getDimensions().z(); k++ )
      for( MeshIndex j = 0; j <= mesh.getDimensions().y(); j++ )
      for( MeshIndex i = 0; i <= mesh.getDimensions().x(); i++ )
      {
         writeInt( format, str, 2 );
         writeInt( format, str, k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i );
         writeInt( format, str, (k+1) * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i );
         if( format == VTK::FileFormat::ascii )
            str << "\n";
      }
   }
};

// 3D grids, vertices
template< typename MeshReal,
          typename Device,
          typename MeshIndex >
struct VTKMeshEntitiesWriter< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, 0 >
{
   using MeshType = Meshes::Grid< 3, MeshReal, Device, MeshIndex >;

   static void exec( const MeshType& mesh, std::ostream& str, VTK::FileFormat format )
   {
      for( MeshIndex k = 0; k < ( mesh.getDimensions().z() + 1 ); k++ )
      for( MeshIndex j = 0; j < ( mesh.getDimensions().y() + 1 ); j++ )
      for( MeshIndex i = 0; i < ( mesh.getDimensions().x() + 1 ); i++ )
      {
         writeInt( format, str, 1 );
         writeInt( format, str, k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i );
         if( format == VTK::FileFormat::ascii )
            str << "\n";
      }
   }
};


// TODO: specialization for disabled entities
template< typename Mesh, int EntityDimension >
struct VTKMeshEntityTypesWriter
{
   static void exec( const Mesh& mesh, std::ostream& str, VTK::FileFormat format )
   {
      using EntityType = typename Mesh::template EntityType< EntityDimension >;
      using Index = typename Mesh::GlobalIndexType;

      const Index entitiesCount = mesh.template getEntitiesCount< EntityType >();
      for( Index i = 0; i < entitiesCount; i++ ) {
         const int type = (int) VTK::TopologyToEntityShape< typename EntityType::EntityTopology >::shape;
         writeInt( format, str, type );
         if( format == VTK::FileFormat::ascii )
            str << "\n";
      }
   }
};

template< int Dimension,
          typename MeshReal,
          typename Device,
          typename MeshIndex,
          int EntityDimension >
struct VTKMeshEntityTypesWriter< Grid< Dimension, MeshReal, Device, MeshIndex >, EntityDimension >
{
   using MeshType = Grid< Dimension, MeshReal, Device, MeshIndex >;

   static void exec( const MeshType& mesh, std::ostream& str, VTK::FileFormat format )
   {
      using EntityType = typename MeshType::template EntityType< EntityDimension >;

      const MeshIndex entitiesCount = mesh.template getEntitiesCount< EntityType >();
      for( MeshIndex i = 0; i < entitiesCount; i++ ) {
         const int type = (int) VTK::GridEntityShape< EntityType >::shape;
         writeInt( format, str, type );
         if( format == VTK::FileFormat::ascii )
            str << "\n";
      }
   }
};

} // namespace detail
} // namespace Writers
} // namespace Meshes
} // namespace TNL
