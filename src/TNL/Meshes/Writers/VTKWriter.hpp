/***************************************************************************
                          VTKWriter.h  -  description
                             -------------------
    begin                : Mar 04, 2017
    copyright            : (C) 2017 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <type_traits>

#include <TNL/Endianness.h>
#include <TNL/Meshes/Writers/VTKWriter.h>
#include <TNL/Meshes/Readers/EntityShape.h>

namespace TNL {
namespace Meshes {
namespace Writers {

namespace __impl {

template< typename T, typename R = void >
struct enable_if_type
{
   using type = R;
};

template< typename T, typename Enable = void >
struct has_entity_topology : std::false_type {};

template< typename T >
struct has_entity_topology< T, typename enable_if_type< typename T::EntityTopology >::type >
: std::true_type
{};


// TODO: 64-bit integers are most likely not supported in the BINARY format
inline void
writeInt( VTKFileFormat format, std::ostream& str, int value )
{
   if( format == VTKFileFormat::BINARY ) {
      value = forceBigEndian( value );
      str.write( reinterpret_cast<const char*>(&value), sizeof(int) );
   }
   else {
      str << value << " ";
   }
}

template< typename Real >
void
writeReal( VTKFileFormat format, std::ostream& str, Real value )
{
   if( format == VTKFileFormat::BINARY ) {
      value = forceBigEndian( value );
      str.write( reinterpret_cast<const char*>(&value), sizeof(Real) );
   }
   else {
      str.precision( std::numeric_limits< Real >::digits10 );
      str << value << " ";
   }
}


template< typename Entity,
          bool _is_mesh_entity = has_entity_topology< Entity >::value >
struct VerticesPerEntity
{
   static constexpr int count = Entity::getVerticesCount();
};

template< typename MeshConfig, typename Device >
struct VerticesPerEntity< MeshEntity< MeshConfig, Device, Topologies::Vertex >, true >
{
   static constexpr int count = 1;
};

template< typename GridEntity >
struct VerticesPerEntity< GridEntity, false >
{
private:
   static constexpr int dim = GridEntity::getEntityDimension();
   static_assert( dim >= 0 && dim <= 3, "unexpected dimension of the grid entity" );

public:
   static constexpr int count =
      (dim == 0) ? 1 :
      (dim == 1) ? 2 :
      (dim == 2) ? 4 :
                   8;
};


template< typename GridEntity >
struct GridEntityShape
{
private:
   static constexpr int dim = GridEntity::getEntityDimension();
   static_assert( dim >= 0 && dim <= 3, "unexpected dimension of the grid entity" );

public:
   static constexpr Readers::EntityShape shape =
      (dim == 0) ? Readers::EntityShape::Vertex :
      (dim == 1) ? Readers::EntityShape::Line :
      (dim == 2) ? Readers::EntityShape::Pixel :
                   Readers::EntityShape::Voxel;
};


template< typename Mesh >
typename Mesh::GlobalIndexType
getAllMeshEntitiesCount( const Mesh& mesh, DimensionTag< 0 > )
{
   using EntityType = typename Mesh::template EntityType< 0 >;
   return mesh.template getEntitiesCount< EntityType >();
}

// TODO: specialization for disabled entities
template< typename Mesh,
          typename DimensionTag = Meshes::DimensionTag< Mesh::getMeshDimension() > >
typename Mesh::GlobalIndexType
getAllMeshEntitiesCount( const Mesh& mesh, DimensionTag = DimensionTag() )
{
   using EntityType = typename Mesh::template EntityType< DimensionTag::value >;
   return mesh.template getEntitiesCount< EntityType >() +
          getAllMeshEntitiesCount( mesh, typename DimensionTag::Decrement() );
}


template< typename Mesh >
typename Mesh::GlobalIndexType
getCellsListSize( const Mesh& mesh, DimensionTag< 0 > )
{
   using EntityType = typename Mesh::template EntityType< 0 >;
   return mesh.template getEntitiesCount< EntityType >() * 2;
}

// TODO: specialization for disabled entities
template< typename Mesh,
          typename DimensionTag = Meshes::DimensionTag< Mesh::getMeshDimension() > >
typename Mesh::GlobalIndexType
getCellsListSize( const Mesh& mesh, DimensionTag = DimensionTag() )
{
   using EntityType = typename Mesh::template EntityType< DimensionTag::value >;
   const auto verticesPerEntity = VerticesPerEntity< EntityType >::count;
   return ( mesh.template getEntitiesCount< EntityType >() * ( verticesPerEntity + 1 ) ) +
          getCellsListSize( mesh, typename DimensionTag::Decrement() );
}


// TODO: specialization for disabled entities
// Unstructured meshes, entities
template< typename Mesh, int EntityDimension >
struct MeshEntitiesVTKWriter
{
   static void exec( const Mesh& mesh, std::ostream& str, VTKFileFormat format )
   {
      using EntityType = typename Mesh::template EntityType< EntityDimension >;
      using Index = typename Mesh::GlobalIndexType;

      const Index entitiesCount = mesh.template getEntitiesCount< EntityType >();
      const Index verticesPerEntity = VerticesPerEntity< EntityType >::count;;
      for( Index i = 0; i < entitiesCount; i++ ) {
         const auto& entity = mesh.template getEntity< EntityType >( i );
         writeInt( format, str, verticesPerEntity );
         for( Index j = 0; j < verticesPerEntity; j++ )
            writeInt( format, str, entity.template getSubentityIndex< 0 >( j ) );
         if( format == VTKFileFormat::ASCII )
            str << "\n";
      }
   }
};

// Unstructured meshes, vertices
template< typename Mesh >
struct MeshEntitiesVTKWriter< Mesh, 0 >
{
   static void exec( const Mesh& mesh, std::ostream& str, VTKFileFormat format )
   {
      using EntityType = typename Mesh::template EntityType< 0 >;
      using Index = typename Mesh::GlobalIndexType;

      const Index entitiesCount = mesh.template getEntitiesCount< EntityType >();
      const Index verticesPerEntity = 1;
      for( Index i = 0; i < entitiesCount; i++ )
      {
         writeInt( format, str, verticesPerEntity );
         writeInt( format, str, i );
         if( format == VTKFileFormat::ASCII )
            str << "\n";
      }
   }
};

// 1D grids, cells
template< typename MeshReal,
          typename Device,
          typename MeshIndex >
struct MeshEntitiesVTKWriter< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, 1 >
{
   using MeshType = Meshes::Grid< 1, MeshReal, Device, MeshIndex >;

   static void exec( const MeshType& mesh, std::ostream& str, VTKFileFormat format )
   {
      for( MeshIndex i = 0; i < mesh.getDimensions().x(); i++ )
      {
         writeInt( format, str, 2 );
         writeInt( format, str, i );
         writeInt( format, str, i+1 );
         if( format == VTKFileFormat::ASCII )
            str << "\n";
      }
   }
};

// 1D grids, vertices
template< typename MeshReal,
          typename Device,
          typename MeshIndex >
struct MeshEntitiesVTKWriter< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, 0 >
{
   using MeshType = Meshes::Grid< 1, MeshReal, Device, MeshIndex >;

   static void exec( const MeshType& mesh, std::ostream& str, VTKFileFormat format )
   {
      for( MeshIndex i = 0; i < mesh.getDimensions().x() + 1; i++ )
      {
         writeInt( format, str, 1 );
         writeInt( format, str, i );
         if( format == VTKFileFormat::ASCII )
            str << "\n";
      }
   }
};

// 2D grids, cells
template< typename MeshReal,
          typename Device,
          typename MeshIndex >
struct MeshEntitiesVTKWriter< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, 2 >
{
   using MeshType = Meshes::Grid< 2, MeshReal, Device, MeshIndex >;

   static void exec( const MeshType& mesh, std::ostream& str, VTKFileFormat format )
   {
      for( MeshIndex j = 0; j < mesh.getDimensions().y(); j++ )
      for( MeshIndex i = 0; i < mesh.getDimensions().x(); i++ )
      {
         writeInt( format, str, 4 );
         writeInt( format, str, j * ( mesh.getDimensions().x() + 1 ) + i );
         writeInt( format, str, j * ( mesh.getDimensions().x() + 1 ) + i + 1 );
         writeInt( format, str, (j+1) * ( mesh.getDimensions().x() + 1 ) + i );
         writeInt( format, str, (j+1) * ( mesh.getDimensions().x() + 1 ) + i + 1 );
         if( format == VTKFileFormat::ASCII )
            str << "\n";
      }
   }
};

// 2D grids, faces
template< typename MeshReal,
          typename Device,
          typename MeshIndex >
struct MeshEntitiesVTKWriter< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, 1 >
{
   using MeshType = Meshes::Grid< 2, MeshReal, Device, MeshIndex >;

   static void exec( const MeshType& mesh, std::ostream& str, VTKFileFormat format )
   {
      for( MeshIndex j = 0; j < mesh.getDimensions().y(); j++ )
      for( MeshIndex i = 0; i < ( mesh.getDimensions().x() + 1 ); i++ )
      {
         writeInt( format, str, 2 );
         writeInt( format, str, j * ( mesh.getDimensions().x() + 1 ) + i );
         writeInt( format, str, (j+1) * ( mesh.getDimensions().x() + 1 ) + i );
         if( format == VTKFileFormat::ASCII )
            str << "\n";
      }

      for( MeshIndex j = 0; j < (mesh.getDimensions().y()+1); j++ )
      for( MeshIndex i = 0; i < mesh.getDimensions().x(); i++ )
      {
         writeInt( format, str, 2 );
         writeInt( format, str, j * ( mesh.getDimensions().x() + 1 ) + i );
         writeInt( format, str, j * ( mesh.getDimensions().x() + 1 ) + i + 1 );
         if( format == VTKFileFormat::ASCII )
            str << "\n";
      }
   }
};

// 2D grids, vertices
template< typename MeshReal,
          typename Device,
          typename MeshIndex >
struct MeshEntitiesVTKWriter< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, 0 >
{
   using MeshType = Meshes::Grid< 2, MeshReal, Device, MeshIndex >;

   static void exec( const MeshType& mesh, std::ostream& str, VTKFileFormat format )
   {
      for( MeshIndex j = 0; j < ( mesh.getDimensions().y() + 1 ); j++ )
      for( MeshIndex i = 0; i < ( mesh.getDimensions().x() + 1 ); i++ )
      {
         writeInt( format, str, 1 );
         writeInt( format, str, j * mesh.getDimensions().x() + i );
         if( format == VTKFileFormat::ASCII )
            str << "\n";
      }
   }
};

// 3D grids, cells
template< typename MeshReal,
          typename Device,
          typename MeshIndex >
struct MeshEntitiesVTKWriter< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, 3 >
{
   using MeshType = Meshes::Grid< 3, MeshReal, Device, MeshIndex >;

   static void exec( const MeshType& mesh, std::ostream& str, VTKFileFormat format )
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
         if( format == VTKFileFormat::ASCII )
            str << "\n";
      }
   }
};

// 3D grids, faces
template< typename MeshReal,
          typename Device,
          typename MeshIndex >
struct MeshEntitiesVTKWriter< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, 2 >
{
   using MeshType = Meshes::Grid< 3, MeshReal, Device, MeshIndex >;

   static void exec( const MeshType& mesh, std::ostream& str, VTKFileFormat format )
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
         if( format == VTKFileFormat::ASCII )
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
         if( format == VTKFileFormat::ASCII )
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
         if( format == VTKFileFormat::ASCII )
            str << "\n";
      }
   }
};

// 3D grids, edges
template< typename MeshReal,
          typename Device,
          typename MeshIndex >
struct MeshEntitiesVTKWriter< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, 1 >
{
   using MeshType = Meshes::Grid< 3, MeshReal, Device, MeshIndex >;

   static void exec( const MeshType& mesh, std::ostream& str, VTKFileFormat format )
   {
      for( MeshIndex k = 0; k <= mesh.getDimensions().z(); k++ )
      for( MeshIndex j = 0; j <= mesh.getDimensions().y(); j++ )
      for( MeshIndex i = 0; i < mesh.getDimensions().x(); i++ )
      {
         writeInt( format, str, 2 );
         writeInt( format, str, k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i );
         writeInt( format, str, k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i + 1 );
         if( format == VTKFileFormat::ASCII )
            str << "\n";
      }

      for( MeshIndex k = 0; k <= mesh.getDimensions().z(); k++ )
      for( MeshIndex j = 0; j < mesh.getDimensions().y(); j++ )
      for( MeshIndex i = 0; i <= mesh.getDimensions().x(); i++ )
      {
         writeInt( format, str, 2 );
         writeInt( format, str, k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i );
         writeInt( format, str, k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + (j+1) * ( mesh.getDimensions().x() + 1 ) + i );
         if( format == VTKFileFormat::ASCII )
            str << "\n";
      }

      for( MeshIndex k = 0; k < mesh.getDimensions().z(); k++ )
      for( MeshIndex j = 0; j <= mesh.getDimensions().y(); j++ )
      for( MeshIndex i = 0; i <= mesh.getDimensions().x(); i++ )
      {
         writeInt( format, str, 2 );
         writeInt( format, str, k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i );
         writeInt( format, str, (k+1) * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i );
         if( format == VTKFileFormat::ASCII )
            str << "\n";
      }
   }
};

// 3D grids, vertices
template< typename MeshReal,
          typename Device,
          typename MeshIndex >
struct MeshEntitiesVTKWriter< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, 0 >
{
   using MeshType = Meshes::Grid< 3, MeshReal, Device, MeshIndex >;

   static void exec( const MeshType& mesh, std::ostream& str, VTKFileFormat format )
   {
      for( MeshIndex k = 0; k < ( mesh.getDimensions().z() + 1 ); k++ )
      for( MeshIndex j = 0; j < ( mesh.getDimensions().y() + 1 ); j++ )
      for( MeshIndex i = 0; i < ( mesh.getDimensions().x() + 1 ); i++ )
      {
         writeInt( format, str, 1 );
         writeInt( format, str, k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i );
         if( format == VTKFileFormat::ASCII )
            str << "\n";
      }
   }
};


// TODO: specialization for disabled entities
template< typename Mesh, int EntityDimension >
struct MeshEntityTypesVTKWriter
{
   static void exec( const Mesh& mesh, std::ostream& str, VTKFileFormat format )
   {
      using EntityType = typename Mesh::template EntityType< EntityDimension >;
      using Index = typename Mesh::GlobalIndexType;

      const Index entitiesCount = mesh.template getEntitiesCount< EntityType >();
      for( Index i = 0; i < entitiesCount; i++ ) {
         const int type = (int) Meshes::Readers::TopologyToEntityShape< typename EntityType::EntityTopology >::shape;
         writeInt( format, str, type );
         if( format == VTKFileFormat::ASCII )
            str << "\n";
      }
   }
};

template< int Dimension,
          typename MeshReal,
          typename Device,
          typename MeshIndex,
          int EntityDimension >
struct MeshEntityTypesVTKWriter< Grid< Dimension, MeshReal, Device, MeshIndex >, EntityDimension >
{
   using MeshType = Grid< Dimension, MeshReal, Device, MeshIndex >;

   static void exec( const MeshType& mesh, std::ostream& str, VTKFileFormat format )
   {
      using EntityType = typename MeshType::template EntityType< EntityDimension >;

      const MeshIndex entitiesCount = mesh.template getEntitiesCount< EntityType >();
      for( MeshIndex i = 0; i < entitiesCount; i++ ) {
         const int type = (int) __impl::GridEntityShape< EntityType >::shape;
         writeInt( format, str, type );
         if( format == VTKFileFormat::ASCII )
            str << "\n";
      }
   }
};

} // namespace __impl

template< typename Mesh >
void
VTKWriter< Mesh >::writeAllEntities( const Mesh& mesh )
{
   writeHeader( mesh );
   writePoints( mesh );

   cellsCount = __impl::getAllMeshEntitiesCount( mesh );
   const IndexType cellsListSize = __impl::getCellsListSize( mesh );

   str << std::endl << "CELLS " << cellsCount << " " << cellsListSize << std::endl;
   Algorithms::TemplateStaticFor< int, 0, Mesh::getMeshDimension() + 1, EntitiesWriter >::exec( mesh, str, format );

   str << std::endl << "CELL_TYPES " << cellsCount << std::endl;
   Algorithms::TemplateStaticFor< int, 0, Mesh::getMeshDimension() + 1, EntityTypesWriter >::exec( mesh, str, format );
}

template< typename Mesh >
   template< int EntityDimension >
void
VTKWriter< Mesh >::writeEntities( const Mesh& mesh )
{
   writeHeader( mesh );
   writePoints( mesh );

   using EntityType = typename Mesh::template EntityType< EntityDimension >;
   cellsCount = mesh.template getEntitiesCount< EntityType >();
   const IndexType verticesPerEntity = __impl::VerticesPerEntity< EntityType >::count;
   const IndexType cellsListSize = cellsCount * ( verticesPerEntity + 1 );

   str << std::endl << "CELLS " << cellsCount << " " << cellsListSize << std::endl;
   EntitiesWriter< EntityDimension >::exec( mesh, str, format );

   str << std::endl << "CELL_TYPES " << cellsCount << std::endl;
   EntityTypesWriter< EntityDimension >::exec( mesh, str, format );
}

template< typename Mesh >
   template< typename Array >
void
VTKWriter< Mesh >::writeDataArray( const Array& array,
                                   const String& name,
                                   const int numberOfComponents,
                                   VTKDataType dataType )
{
   // use a host buffer if direct access to the array elements is not possible
   if( std::is_same< typename Array::DeviceType, Devices::Cuda >::value )
   {
      using HostArray = typename Array::template Self< typename Array::ValueType, Devices::Host >;
      HostArray hostBuffer;
      hostBuffer = array;
      writeDataArray( hostBuffer, name, numberOfComponents, dataType );
      return;
   }

   if( numberOfComponents != 1 && numberOfComponents != 3 )
      throw std::logic_error("Unsupported numberOfComponents parameter: " + std::to_string(numberOfComponents));

   if( dataType == VTKDataType::CellData )
      if( array.getSize() / numberOfComponents != cellsCount )
         throw std::length_error("Mismatched array size for CELL_DATA section: " + std::to_string(array.getSize())
                                 + " (there are " + std::to_string(cellsCount) + " cells in the file)");
   if( dataType == VTKDataType::PointData )
      if( array.getSize() / numberOfComponents != pointsCount )
         throw std::length_error("Mismatched array size for POINT_DATA section: " + std::to_string(array.getSize())
                                 + " (there are " + std::to_string(pointsCount) + " points in the file)");

   // check that we won't start the section second time
   if( dataType != currentSection && cellDataArrays * pointDataArrays != 0 )
      throw std::logic_error("The requested data section is not the current section and it has already been written.");

   // start the appropriate section if necessary
   if( dataType == VTKDataType::CellData && cellDataArrays == 0 ) {
      str << std::endl << "CELL_DATA " << cellsCount << std::endl;
      ++cellDataArrays;
   }
   if( dataType == VTKDataType::PointData && pointDataArrays == 0 ) {
      str << std::endl << "POINT_DATA " << pointsCount << std::endl;
      ++pointDataArrays;
   }

   // write DataArray header
   if( numberOfComponents == 1 ) {
      str << "SCALARS " << name << " " << getType< typename Array::ValueType >() << " 1" << std::endl;
      str << "LOOKUP_TABLE default" << std::endl;
   }
   else {
      str << "VECTORS " << name << " " << getType< typename Array::ValueType >() << " 1" << std::endl;
   }

   using Meshes::Writers::__impl::writeReal;
   for( IndexType i = 0; i < array.getSize(); i++ ) {
      writeReal( format, str, array[i] );
      if( format == Meshes::Writers::VTKFileFormat::ASCII )
         str << "\n";
   }
}

template< typename Mesh >
void
VTKWriter< Mesh >::writeHeader( const Mesh& mesh )
{
    str << "# vtk DataFile Version 2.0\n"
        << "TNL DATA\n"
        << ((format == VTKFileFormat::ASCII) ? "ASCII\n" : "BINARY\n")
        << "DATASET UNSTRUCTURED_GRID\n";
}

template< typename Mesh >
void
VTKWriter< Mesh >::writePoints( const Mesh& mesh )
{
   using __impl::writeReal;
   pointsCount = mesh.template getEntitiesCount< typename Mesh::Vertex >();
   str << "POINTS " << pointsCount << " " << getType< typename Mesh::RealType >() << std::endl;
   for( IndexType i = 0; i < pointsCount; i++ ) {
      const auto& vertex = mesh.template getEntity< typename Mesh::Vertex >( i );
      const auto& point = vertex.getPoint();
      for( IndexType j = 0; j < point.getSize(); j++ )
         writeReal( format, str, point[ j ] );
      // VTK needs zeros for unused dimensions
      for( IndexType j = 0; j < 3 - point.getSize(); j++ )
         writeReal( format, str, (typename Mesh::PointType::RealType) 0 );
      if( format == VTKFileFormat::ASCII )
         str << "\n";
   }
}

} // namespace Writers
} // namespace Meshes
} // namespace TNL
