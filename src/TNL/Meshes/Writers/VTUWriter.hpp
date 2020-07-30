/***************************************************************************
                          VTUWriter.hpp  -  description
                             -------------------
    begin                : Mar 18, 2020
    copyright            : (C) 2020 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovský

#pragma once

#include <limits>

#include <TNL/Meshes/Writers/VTUWriter.h>
#include <TNL/Meshes/Writers/VerticesPerEntity.h>
#include <TNL/Endianness.h>
#include <TNL/base64.h>
#ifdef HAVE_ZLIB
   #include <TNL/zlib_compression.h>
#endif

namespace TNL {
namespace Meshes {
namespace Writers {

namespace details {

// TODO: specialization for disabled entities
// Unstructured meshes, entities
template< typename Mesh, int EntityDimension >
struct MeshEntitiesVTUCollector
{
   static void exec( const Mesh& mesh,
                     std::vector< typename Mesh::GlobalIndexType > & connectivity,
                     std::vector< typename Mesh::GlobalIndexType > & offsets,
                     std::vector< std::uint8_t > & types )
   {
      using EntityType = typename Mesh::template EntityType< EntityDimension >;
      using Index = typename Mesh::GlobalIndexType;

      const Index entitiesCount = mesh.template getEntitiesCount< EntityType >();
      const Index verticesPerEntity = VerticesPerEntity< EntityType >::count;;
      for( Index i = 0; i < entitiesCount; i++ ) {
         const auto& entity = mesh.template getEntity< EntityType >( i );
         for( Index j = 0; j < verticesPerEntity; j++ )
            connectivity.push_back( entity.template getSubentityIndex< 0 >( j ) );
         offsets.push_back( connectivity.size() );
         const std::uint8_t type = (std::uint8_t) VTK::TopologyToEntityShape< typename EntityType::EntityTopology >::shape;
         types.push_back( type );
      }
   }
};

// Unstructured meshes, vertices
template< typename Mesh >
struct MeshEntitiesVTUCollector< Mesh, 0 >
{
   static void exec( const Mesh& mesh,
                     std::vector< typename Mesh::GlobalIndexType > & connectivity,
                     std::vector< typename Mesh::GlobalIndexType > & offsets,
                     std::vector< std::uint8_t > & types )
   {
      using EntityType = typename Mesh::template EntityType< 0 >;
      using Index = typename Mesh::GlobalIndexType;

      const Index entitiesCount = mesh.template getEntitiesCount< EntityType >();
      for( Index i = 0; i < entitiesCount; i++ ) {
         connectivity.push_back( i );
         offsets.push_back( connectivity.size() );
         const std::uint8_t type = (std::uint8_t) VTK::TopologyToEntityShape< typename EntityType::EntityTopology >::shape;
         types.push_back( type );
      }
   }
};

// 1D grids, cells
template< typename MeshReal,
          typename Device,
          typename MeshIndex >
struct MeshEntitiesVTUCollector< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, 1 >
{
   using Mesh = Meshes::Grid< 1, MeshReal, Device, MeshIndex >;

   static void exec( const Mesh& mesh,
                     std::vector< typename Mesh::GlobalIndexType > & connectivity,
                     std::vector< typename Mesh::GlobalIndexType > & offsets,
                     std::vector< std::uint8_t > & types )
   {
      for( MeshIndex i = 0; i < mesh.getDimensions().x(); i++ )
      {
         connectivity.push_back( i );
         connectivity.push_back( i+1 );
         offsets.push_back( connectivity.size() );
         types.push_back( (std::uint8_t) VTK::EntityShape::Line );
      }
   }
};

// 1D grids, vertices
template< typename MeshReal,
          typename Device,
          typename MeshIndex >
struct MeshEntitiesVTUCollector< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, 0 >
{
   using Mesh = Meshes::Grid< 1, MeshReal, Device, MeshIndex >;

   static void exec( const Mesh& mesh,
                     std::vector< typename Mesh::GlobalIndexType > & connectivity,
                     std::vector< typename Mesh::GlobalIndexType > & offsets,
                     std::vector< std::uint8_t > & types )
   {
      for( MeshIndex i = 0; i < mesh.getDimensions().x() + 1; i++ )
      {
         connectivity.push_back( i );
         offsets.push_back( connectivity.size() );
         types.push_back( (std::uint8_t) VTK::EntityShape::Vertex );
      }
   }
};

// 2D grids, cells
template< typename MeshReal,
          typename Device,
          typename MeshIndex >
struct MeshEntitiesVTUCollector< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, 2 >
{
   using Mesh = Meshes::Grid< 2, MeshReal, Device, MeshIndex >;

   static void exec( const Mesh& mesh,
                     std::vector< typename Mesh::GlobalIndexType > & connectivity,
                     std::vector< typename Mesh::GlobalIndexType > & offsets,
                     std::vector< std::uint8_t > & types )
   {
      for( MeshIndex j = 0; j < mesh.getDimensions().y(); j++ )
      for( MeshIndex i = 0; i < mesh.getDimensions().x(); i++ )
      {
         connectivity.push_back( j * ( mesh.getDimensions().x() + 1 ) + i );
         connectivity.push_back( j * ( mesh.getDimensions().x() + 1 ) + i + 1 );
         connectivity.push_back( (j+1) * ( mesh.getDimensions().x() + 1 ) + i );
         connectivity.push_back( (j+1) * ( mesh.getDimensions().x() + 1 ) + i + 1 );
         offsets.push_back( connectivity.size() );
         types.push_back( (std::uint8_t) VTK::EntityShape::Pixel );
      }
   }
};

// 2D grids, faces
template< typename MeshReal,
          typename Device,
          typename MeshIndex >
struct MeshEntitiesVTUCollector< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, 1 >
{
   using Mesh = Meshes::Grid< 2, MeshReal, Device, MeshIndex >;

   static void exec( const Mesh& mesh,
                     std::vector< typename Mesh::GlobalIndexType > & connectivity,
                     std::vector< typename Mesh::GlobalIndexType > & offsets,
                     std::vector< std::uint8_t > & types )
   {
      for( MeshIndex j = 0; j < mesh.getDimensions().y(); j++ )
      for( MeshIndex i = 0; i < ( mesh.getDimensions().x() + 1 ); i++ )
      {
         connectivity.push_back( j * ( mesh.getDimensions().x() + 1 ) + i );
         connectivity.push_back( (j+1) * ( mesh.getDimensions().x() + 1 ) + i );
         offsets.push_back( connectivity.size() );
         types.push_back( (std::uint8_t) VTK::EntityShape::Line );
      }

      for( MeshIndex j = 0; j < (mesh.getDimensions().y()+1); j++ )
      for( MeshIndex i = 0; i < mesh.getDimensions().x(); i++ )
      {
         connectivity.push_back( j * ( mesh.getDimensions().x() + 1 ) + i );
         connectivity.push_back( j * ( mesh.getDimensions().x() + 1 ) + i + 1 );
         offsets.push_back( connectivity.size() );
         types.push_back( (std::uint8_t) VTK::EntityShape::Line );
      }
   }
};

// 2D grids, vertices
template< typename MeshReal,
          typename Device,
          typename MeshIndex >
struct MeshEntitiesVTUCollector< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, 0 >
{
   using Mesh = Meshes::Grid< 2, MeshReal, Device, MeshIndex >;

   static void exec( const Mesh& mesh,
                     std::vector< typename Mesh::GlobalIndexType > & connectivity,
                     std::vector< typename Mesh::GlobalIndexType > & offsets,
                     std::vector< std::uint8_t > & types )
   {
      for( MeshIndex j = 0; j < ( mesh.getDimensions().y() + 1 ); j++ )
      for( MeshIndex i = 0; i < ( mesh.getDimensions().x() + 1 ); i++ )
      {
         connectivity.push_back( j * mesh.getDimensions().x() + i );
         offsets.push_back( connectivity.size() );
         types.push_back( (std::uint8_t) VTK::EntityShape::Vertex );
      }
   }
};

// 3D grids, cells
template< typename MeshReal,
          typename Device,
          typename MeshIndex >
struct MeshEntitiesVTUCollector< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, 3 >
{
   using Mesh = Meshes::Grid< 3, MeshReal, Device, MeshIndex >;

   static void exec( const Mesh& mesh,
                     std::vector< typename Mesh::GlobalIndexType > & connectivity,
                     std::vector< typename Mesh::GlobalIndexType > & offsets,
                     std::vector< std::uint8_t > & types )
   {
      for( MeshIndex k = 0; k < mesh.getDimensions().z(); k++ )
      for( MeshIndex j = 0; j < mesh.getDimensions().y(); j++ )
      for( MeshIndex i = 0; i < mesh.getDimensions().x(); i++ )
      {
         connectivity.push_back( k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i );
         connectivity.push_back( k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i + 1 );
         connectivity.push_back( k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + (j+1) * ( mesh.getDimensions().x() + 1 ) + i );
         connectivity.push_back( k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + (j+1) * ( mesh.getDimensions().x() + 1 ) + i + 1 );
         connectivity.push_back( (k+1) * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i );
         connectivity.push_back( (k+1) * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i + 1 );
         connectivity.push_back( (k+1) * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + (j+1) * ( mesh.getDimensions().x() + 1 ) + i );
         connectivity.push_back( (k+1) * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + (j+1) * ( mesh.getDimensions().x() + 1 ) + i + 1 );
         offsets.push_back( connectivity.size() );
         types.push_back( (std::uint8_t) VTK::EntityShape::Voxel );
      }
   }
};

// 3D grids, faces
template< typename MeshReal,
          typename Device,
          typename MeshIndex >
struct MeshEntitiesVTUCollector< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, 2 >
{
   using Mesh = Meshes::Grid< 3, MeshReal, Device, MeshIndex >;

   static void exec( const Mesh& mesh,
                     std::vector< typename Mesh::GlobalIndexType > & connectivity,
                     std::vector< typename Mesh::GlobalIndexType > & offsets,
                     std::vector< std::uint8_t > & types )
   {
      for( MeshIndex k = 0; k < mesh.getDimensions().z(); k++ )
      for( MeshIndex j = 0; j < mesh.getDimensions().y(); j++ )
      for( MeshIndex i = 0; i <= mesh.getDimensions().x(); i++ )
      {
         connectivity.push_back( k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i );
         connectivity.push_back( k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + (j+1) * ( mesh.getDimensions().x() + 1 ) + i );
         connectivity.push_back( (k+1) * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i );
         connectivity.push_back( (k+1) * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + (j+1) * ( mesh.getDimensions().x() + 1 ) + i );
         offsets.push_back( connectivity.size() );
         types.push_back( (std::uint8_t) VTK::EntityShape::Pixel );
      }

      for( MeshIndex k = 0; k < mesh.getDimensions().z(); k++ )
      for( MeshIndex j = 0; j <= mesh.getDimensions().y(); j++ )
      for( MeshIndex i = 0; i < mesh.getDimensions().x(); i++ )
      {
         connectivity.push_back( k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i );
         connectivity.push_back( k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i + 1 );
         connectivity.push_back( (k+1) * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i );
         connectivity.push_back( (k+1) * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i + 1 );
         offsets.push_back( connectivity.size() );
         types.push_back( (std::uint8_t) VTK::EntityShape::Pixel );
      }

      for( MeshIndex k = 0; k <= mesh.getDimensions().z(); k++ )
      for( MeshIndex j = 0; j < mesh.getDimensions().y(); j++ )
      for( MeshIndex i = 0; i < mesh.getDimensions().x(); i++ )
      {
         connectivity.push_back( k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i );
         connectivity.push_back( k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i + 1 );
         connectivity.push_back( k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + (j+1) * ( mesh.getDimensions().x() + 1 ) + i );
         connectivity.push_back( k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + (j+1) * ( mesh.getDimensions().x() + 1 ) + i + 1 );
         offsets.push_back( connectivity.size() );
         types.push_back( (std::uint8_t) VTK::EntityShape::Pixel );
      }
   }
};

// 3D grids, edges
template< typename MeshReal,
          typename Device,
          typename MeshIndex >
struct MeshEntitiesVTUCollector< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, 1 >
{
   using Mesh = Meshes::Grid< 3, MeshReal, Device, MeshIndex >;

   static void exec( const Mesh& mesh,
                     std::vector< typename Mesh::GlobalIndexType > & connectivity,
                     std::vector< typename Mesh::GlobalIndexType > & offsets,
                     std::vector< std::uint8_t > & types )
   {
      for( MeshIndex k = 0; k <= mesh.getDimensions().z(); k++ )
      for( MeshIndex j = 0; j <= mesh.getDimensions().y(); j++ )
      for( MeshIndex i = 0; i < mesh.getDimensions().x(); i++ )
      {
         connectivity.push_back( k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i );
         connectivity.push_back( k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i + 1 );
         offsets.push_back( connectivity.size() );
         types.push_back( (std::uint8_t) VTK::EntityShape::Line );
      }

      for( MeshIndex k = 0; k <= mesh.getDimensions().z(); k++ )
      for( MeshIndex j = 0; j < mesh.getDimensions().y(); j++ )
      for( MeshIndex i = 0; i <= mesh.getDimensions().x(); i++ )
      {
         connectivity.push_back( k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i );
         connectivity.push_back( k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + (j+1) * ( mesh.getDimensions().x() + 1 ) + i );
         offsets.push_back( connectivity.size() );
         types.push_back( (std::uint8_t) VTK::EntityShape::Line );
      }

      for( MeshIndex k = 0; k < mesh.getDimensions().z(); k++ )
      for( MeshIndex j = 0; j <= mesh.getDimensions().y(); j++ )
      for( MeshIndex i = 0; i <= mesh.getDimensions().x(); i++ )
      {
         connectivity.push_back( k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i );
         connectivity.push_back( (k+1) * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i );
         offsets.push_back( connectivity.size() );
         types.push_back( (std::uint8_t) VTK::EntityShape::Line );
      }
   }
};

// 3D grids, vertices
template< typename MeshReal,
          typename Device,
          typename MeshIndex >
struct MeshEntitiesVTUCollector< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, 0 >
{
   using Mesh = Meshes::Grid< 3, MeshReal, Device, MeshIndex >;

   static void exec( const Mesh& mesh,
                     std::vector< typename Mesh::GlobalIndexType > & connectivity,
                     std::vector< typename Mesh::GlobalIndexType > & offsets,
                     std::vector< std::uint8_t > & types )
   {
      for( MeshIndex k = 0; k < ( mesh.getDimensions().z() + 1 ); k++ )
      for( MeshIndex j = 0; j < ( mesh.getDimensions().y() + 1 ); j++ )
      for( MeshIndex i = 0; i < ( mesh.getDimensions().x() + 1 ); i++ )
      {
         connectivity.push_back( k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i );
         offsets.push_back( connectivity.size() );
         types.push_back( (std::uint8_t) VTK::EntityShape::Vertex );
      }
   }
};

} // namespace details

template< typename Mesh >
void
VTUWriter< Mesh >::writeMetadata( int cycle, double time )
{
   if( ! vtkfileOpen )
      writeHeader();

   if( cycle >= 0 || time >= 0 )
      str << "<FieldData>\n";

   if( cycle >= 0 ) {
      str << "<DataArray type=\"Int32\" Name=\"CYCLE\" NumberOfTuples=\"1\" format=\"ascii\">"
          << cycle << "</DataArray>\n";
   }
   if( time >= 0 ) {
      str.precision( std::numeric_limits< double >::digits10 );
      str << "<DataArray type=\"Float64\" Name=\"TIME\" NumberOfTuples=\"1\" format=\"ascii\">"
          << time << "</DataArray>\n";
   }

   if( cycle >= 0 || time >= 0 )
      str << "</FieldData>\n";
}

template< typename Mesh >
   template< int EntityDimension >
void
VTUWriter< Mesh >::writeEntities( const Mesh& mesh )
{
   // count points and cells before any writing
   pointsCount = mesh.template getEntitiesCount< typename Mesh::Vertex >();
   using EntityType = typename Mesh::template EntityType< EntityDimension >;
   cellsCount = mesh.template getEntitiesCount< EntityType >();

   if( ! vtkfileOpen )
      writeHeader();
   closePiece();
   str << "<Piece NumberOfPoints=\"" << pointsCount << "\" NumberOfCells=\"" << cellsCount << "\">\n";
   pieceOpen = true;

   // write points
   writePoints( mesh );

   // collect all data before writing
   std::vector< IndexType > connectivity, offsets;
   std::vector< std::uint8_t > types;
   EntitiesCollector< EntityDimension >::exec( mesh, connectivity, offsets, types );

   // create array views that can be passed to writeDataArray
   Containers::ArrayView< IndexType > connectivity_v( connectivity.data(), connectivity.size() );
   Containers::ArrayView< IndexType > offsets_v( offsets.data(), offsets.size() );
   Containers::ArrayView< std::uint8_t > types_v( types.data(), types.size() );

   // write cells
   str << "<Cells>\n";
   writeDataArray( connectivity_v, "connectivity", 0 );
   writeDataArray( offsets_v, "offsets", 0 );
   writeDataArray( types_v, "types", 0 );
   str << "</Cells>\n";
}

template< typename Mesh >
   template< typename Array >
void
VTUWriter< Mesh >::writePointData( const Array& array,
                                   const String& name,
                                   const int numberOfComponents )
{
   if( ! pieceOpen )
      throw std::logic_error("The <Piece> tag has not been opened yet - call writeEntities first.");
   if( array.getSize() / numberOfComponents != pointsCount )
      throw std::length_error("Mismatched array size for <PointData> section: " + std::to_string(array.getSize())
                              + " (there are " + std::to_string(pointsCount) + " points in the file)");
   openPointData();
   writeDataArray( array, name, numberOfComponents );
}

template< typename Mesh >
   template< typename Array >
void
VTUWriter< Mesh >::writeCellData( const Array& array,
                                  const String& name,
                                  const int numberOfComponents )
{
   if( ! pieceOpen )
      throw std::logic_error("The <Piece> tag has not been opened yet - call writeEntities first.");
   if( array.getSize() / numberOfComponents != cellsCount )
      throw std::length_error("Mismatched array size for <CellData> section: " + std::to_string(array.getSize())
                              + " (there are " + std::to_string(cellsCount) + " cells in the file)");
   openCellData();
   writeDataArray( array, name, numberOfComponents );
}

template< typename Mesh >
   template< typename Array >
void
VTUWriter< Mesh >::writeDataArray( const Array& array,
                                   const String& name,
                                   const int numberOfComponents )
{
   // use a host buffer if direct access to the array elements is not possible
   if( std::is_same< typename Array::DeviceType, Devices::Cuda >::value )
   {
      using HostArray = typename Array::template Self< typename Array::ValueType, Devices::Host >;
      HostArray hostBuffer;
      hostBuffer = array;
      writeDataArray( hostBuffer, name, numberOfComponents );
      return;
   }

   if( numberOfComponents != 0 && numberOfComponents != 1 && numberOfComponents != 3 )
      throw std::logic_error("Unsupported numberOfComponents parameter: " + std::to_string(numberOfComponents));

   // write DataArray header
   str << "<DataArray type=\"" << VTK::getTypeName( array[0] ) << "\"";
   str << " Name=\"" << name << "\"";
   if( numberOfComponents > 0 )
      str << " NumberOfComponents=\"" << numberOfComponents << "\"";
   str << " format=\"" << ((format == VTK::FileFormat::ascii) ? "ascii" : "binary") << "\">\n";

   switch( format )
   {
      case VTK::FileFormat::ascii:
         str.precision( std::numeric_limits< typename Array::ValueType >::digits10 );
         for( IndexType i = 0; i < array.getSize(); i++ )
            // If Array::ValueType is uint8_t, it might be a typedef for unsigned char, which
            // would be normally printed as char rather than a number. Hence, we use the trick
            // with unary operator+, see https://stackoverflow.com/a/28414758
            str << +array[i] << " ";
         str << "\n";
         break;
      case VTK::FileFormat::binary:
         write_encoded_block< HeaderType >( array.getData(), array.getSize(), str );
         str << "\n";
         break;
      case VTK::FileFormat::zlib_compressed:
#ifdef HAVE_ZLIB
         write_compressed_block< HeaderType >( array.getData(), array.getSize(), str );
         str << "\n";
         break;
#else
         throw std::runtime_error("The ZLIB compression algorithm is not available in this build. Please recompile the program with -DHAVE_ZLIB.");
#endif
   }

   // write DataArray footer
   str << "</DataArray>\n";
}

template< typename Mesh >
void
VTUWriter< Mesh >::writePoints( const Mesh& mesh )
{
   // copy all coordinates into a contiguous array
   using BufferType = Containers::Array< MeshRealType, Devices::Host, IndexType >;
   BufferType buffer( 3 * pointsCount );
   IndexType k = 0;
   for( IndexType i = 0; i < pointsCount; i++ ) {
      const auto& vertex = mesh.template getEntity< typename Mesh::Vertex >( i );
      const auto& point = vertex.getPoint();
      for( IndexType j = 0; j < point.getSize(); j++ )
         buffer[ k++ ] = point[ j ];
      // VTK needs zeros for unused dimensions
      for( IndexType j = point.getSize(); j < 3; j++ )
         buffer[ k++ ] = 0;
   }

   // write the buffer
   str << "<Points>\n";
   writeDataArray( buffer, "Points", 3 );
   str << "</Points>\n";
}

template< typename Mesh >
void
VTUWriter< Mesh >::writeHeader()
{
   str << "<?xml version=\"1.0\"?>\n";
   str << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\"";
   if( isLittleEndian() )
      str << " byte_order=\"LittleEndian\"";
   else
      str << " byte_order=\"BigEndian\"";
   str << " header_type=\"" << VTK::getTypeName( HeaderType{} ) << "\"";
   if( format == VTK::FileFormat::zlib_compressed )
      str << " compressor=\"vtkZLibDataCompressor\"";
   str << ">\n";
   str << "<UnstructuredGrid>\n";

   vtkfileOpen = true;
}

template< typename Mesh >
void
VTUWriter< Mesh >::writeFooter()
{
   closePiece();
   str << "</UnstructuredGrid>\n";
   str << "</VTKFile>\n";
}

template< typename Mesh >
VTUWriter< Mesh >::~VTUWriter()
{
   if( vtkfileOpen )
      writeFooter();
}

template< typename Mesh >
void
VTUWriter< Mesh >::openCellData()
{
   if( cellDataClosed )
      throw std::logic_error("The <CellData> tag has already been closed in the current <Piece> section.");
   closePointData();
   if( ! cellDataOpen ) {
      str << "<CellData>\n";
      cellDataOpen = true;
   }
}

template< typename Mesh >
void
VTUWriter< Mesh >::closeCellData()
{
   if( cellDataOpen ) {
      str << "</CellData>\n";
      cellDataClosed = true;
      cellDataOpen = false;
   }
}

template< typename Mesh >
void
VTUWriter< Mesh >::openPointData()
{
   if( pointDataClosed )
      throw std::logic_error("The <PointData> tag has already been closed in the current <Piece> section.");
   closeCellData();
   if( ! pointDataOpen ) {
      str << "<PointData>\n";
      pointDataOpen = true;
   }
}

template< typename Mesh >
void
VTUWriter< Mesh >::closePointData()
{
   if( pointDataOpen ) {
      str << "</PointData>\n";
      pointDataClosed = true;
      pointDataOpen = false;
   }
}

template< typename Mesh >
void
VTUWriter< Mesh >::closePiece()
{
   if( pieceOpen ) {
      closeCellData();
      closePointData();
      str << "</Piece>\n";

      // reset indicators - new <Piece> can be started
      pieceOpen = false;
      cellDataOpen = cellDataClosed = false;
      pointDataOpen = pointDataClosed = false;
   }
}

} // namespace Writers
} // namespace Meshes
} // namespace TNL
