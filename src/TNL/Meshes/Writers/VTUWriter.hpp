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
#include <TNL/Meshes/Grid.h>
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
      for( Index i = 0; i < entitiesCount; i++ ) {
         const auto& entity = mesh.template getEntity< EntityType >( i );
         const Index verticesPerEntity = entity.template getSubentitiesCount< 0 >();
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
   using Entity = typename Mesh::template EntityType< 1 >;

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
         types.push_back( (std::uint8_t) VTK::GridEntityShape< Entity >::shape );
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
   using Entity = typename Mesh::template EntityType< 0 >;

   static void exec( const Mesh& mesh,
                     std::vector< typename Mesh::GlobalIndexType > & connectivity,
                     std::vector< typename Mesh::GlobalIndexType > & offsets,
                     std::vector< std::uint8_t > & types )
   {
      for( MeshIndex i = 0; i < mesh.getDimensions().x() + 1; i++ )
      {
         connectivity.push_back( i );
         offsets.push_back( connectivity.size() );
         types.push_back( (std::uint8_t) VTK::GridEntityShape< Entity >::shape );
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
   using Entity = typename Mesh::template EntityType< 2 >;

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
         types.push_back( (std::uint8_t) VTK::GridEntityShape< Entity >::shape );
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
   using Entity = typename Mesh::template EntityType< 1 >;

   static void exec( const Mesh& mesh,
                     std::vector< typename Mesh::GlobalIndexType > & connectivity,
                     std::vector< typename Mesh::GlobalIndexType > & offsets,
                     std::vector< std::uint8_t > & types )
   {
      for( MeshIndex j = 0; j < mesh.getDimensions().y(); j++ )
      for( MeshIndex i = 0; i < (mesh.getDimensions().x() + 1); i++ )
      {
         connectivity.push_back( j * ( mesh.getDimensions().x() + 1 ) + i );
         connectivity.push_back( (j+1) * ( mesh.getDimensions().x() + 1 ) + i );
         offsets.push_back( connectivity.size() );
         types.push_back( (std::uint8_t) VTK::GridEntityShape< Entity >::shape );
      }

      for( MeshIndex j = 0; j < (mesh.getDimensions().y() + 1); j++ )
      for( MeshIndex i = 0; i < mesh.getDimensions().x(); i++ )
      {
         connectivity.push_back( j * ( mesh.getDimensions().x() + 1 ) + i );
         connectivity.push_back( j * ( mesh.getDimensions().x() + 1 ) + i + 1 );
         offsets.push_back( connectivity.size() );
         types.push_back( (std::uint8_t) VTK::GridEntityShape< Entity >::shape );
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
   using Entity = typename Mesh::template EntityType< 0 >;

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
         types.push_back( (std::uint8_t) VTK::GridEntityShape< Entity >::shape );
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
   using Entity = typename Mesh::template EntityType< 3 >;

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
         types.push_back( (std::uint8_t) VTK::GridEntityShape< Entity >::shape );
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
   using Entity = typename Mesh::template EntityType< 2 >;

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
         types.push_back( (std::uint8_t) VTK::GridEntityShape< Entity >::shape );
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
         types.push_back( (std::uint8_t) VTK::GridEntityShape< Entity >::shape );
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
         types.push_back( (std::uint8_t) VTK::GridEntityShape< Entity >::shape );
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
   using Entity = typename Mesh::template EntityType< 1 >;

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
         types.push_back( (std::uint8_t) VTK::GridEntityShape< Entity >::shape );
      }

      for( MeshIndex k = 0; k <= mesh.getDimensions().z(); k++ )
      for( MeshIndex j = 0; j < mesh.getDimensions().y(); j++ )
      for( MeshIndex i = 0; i <= mesh.getDimensions().x(); i++ )
      {
         connectivity.push_back( k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i );
         connectivity.push_back( k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + (j+1) * ( mesh.getDimensions().x() + 1 ) + i );
         offsets.push_back( connectivity.size() );
         types.push_back( (std::uint8_t) VTK::GridEntityShape< Entity >::shape );
      }

      for( MeshIndex k = 0; k < mesh.getDimensions().z(); k++ )
      for( MeshIndex j = 0; j <= mesh.getDimensions().y(); j++ )
      for( MeshIndex i = 0; i <= mesh.getDimensions().x(); i++ )
      {
         connectivity.push_back( k * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i );
         connectivity.push_back( (k+1) * ( mesh.getDimensions().y() + 1 ) * ( mesh.getDimensions().x() + 1 ) + j * ( mesh.getDimensions().x() + 1 ) + i );
         offsets.push_back( connectivity.size() );
         types.push_back( (std::uint8_t) VTK::GridEntityShape< Entity >::shape );
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
   using Entity = typename Mesh::template EntityType< 0 >;

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
         types.push_back( (std::uint8_t) VTK::GridEntityShape< Entity >::shape );
      }
   }
};

// specialization for meshes
template< typename Mesh >
struct PolyhedralFacesWriter
{
   // specialization for all meshes except polyhedral
   template< typename W, typename M >
   static std::enable_if_t< ! std::is_same< typename M::Config::CellTopology, Topologies::Polyhedron >::value >
   exec( W& writer, const M& mesh )
   {}

   // specialization for polyhedral meshes
   template< typename W, typename M >
   static std::enable_if_t< std::is_same< typename M::Config::CellTopology, Topologies::Polyhedron >::value >
   exec( W& writer, const M& mesh )
   {
      // build the "face stream" for VTK
      using IndexType = typename Mesh::GlobalIndexType;
      std::vector< IndexType > faces, faceoffsets;
      for( IndexType c = 0; c < mesh.template getEntitiesCount< M::getMeshDimension() >(); c++ ) {
         const IndexType num_faces = mesh.template getSubentitiesCount< M::getMeshDimension(), M::getMeshDimension() - 1 >( c );
         faces.push_back( num_faces );
         for( IndexType f = 0; f < num_faces; f++ ) {
            const auto& face = mesh.template getEntity< M::getMeshDimension() - 1 >( mesh.template getSubentityIndex< M::getMeshDimension(), M::getMeshDimension() - 1 >( c, f ) );
            const IndexType num_vertices = face.template getSubentitiesCount< 0 >();
            faces.push_back( num_vertices );
            for( IndexType v = 0; v < num_vertices; v++ ) {
               const IndexType vertex = face.template getSubentityIndex< 0 >( v );
               faces.push_back( vertex );
            }
         }
         faceoffsets.push_back( faces.size() );
      }

      // create array views that can be passed to writeDataArray
      Containers::ArrayView< IndexType, Devices::Host, std::uint64_t > faces_v( faces.data(), faces.size() );
      Containers::ArrayView< IndexType, Devices::Host, std::uint64_t > faceoffsets_v( faceoffsets.data(), faceoffsets.size() );

      // write cells
      writer.writeDataArray( faces_v, "faces", 0 );
      writer.writeDataArray( faceoffsets_v, "faceoffsets", 0 );
   }
};

// specialization for grids
template< int Dimension,
          typename MeshReal,
          typename Device,
          typename MeshIndex >
struct PolyhedralFacesWriter< Meshes::Grid< Dimension, MeshReal, Device, MeshIndex > >
{
   template< typename W, typename M >
   static void exec( W& writer, const M& mesh )
   {}
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
   using IndexType = typename Mesh::GlobalIndexType;
   std::vector< IndexType > connectivity, offsets;
   std::vector< std::uint8_t > types;
   EntitiesCollector< EntityDimension >::exec( mesh, connectivity, offsets, types );

   // create array views that can be passed to writeDataArray
   Containers::ArrayView< IndexType, Devices::Host, std::uint64_t > connectivity_v( connectivity.data(), connectivity.size() );
   Containers::ArrayView< IndexType, Devices::Host, std::uint64_t > offsets_v( offsets.data(), offsets.size() );
   Containers::ArrayView< std::uint8_t, Devices::Host, std::uint64_t > types_v( types.data(), types.size() );

   // write cells
   str << "<Cells>\n";
   writeDataArray( connectivity_v, "connectivity", 0 );
   writeDataArray( offsets_v, "offsets", 0 );
   writeDataArray( types_v, "types", 0 );
   // write faces if the mesh is polyhedral
   details::PolyhedralFacesWriter< Mesh >::exec( *this, mesh );
   str << "</Cells>\n";
}

template< typename Mesh >
   template< typename Array >
void
VTUWriter< Mesh >::writePointData( const Array& array,
                                   const std::string& name,
                                   const int numberOfComponents )
{
   if( ! pieceOpen )
      throw std::logic_error("The <Piece> tag has not been opened yet - call writeEntities first.");
   if( array.getSize() / numberOfComponents != typename Array::IndexType(pointsCount) )
      throw std::length_error("Mismatched array size for <PointData> section: " + std::to_string(array.getSize())
                              + " (there are " + std::to_string(pointsCount) + " points in the file)");
   openPointData();
   writeDataArray( array, name, numberOfComponents );
}

template< typename Mesh >
   template< typename Array >
void
VTUWriter< Mesh >::writeCellData( const Array& array,
                                  const std::string& name,
                                  const int numberOfComponents )
{
   if( ! pieceOpen )
      throw std::logic_error("The <Piece> tag has not been opened yet - call writeEntities first.");
   if( array.getSize() / numberOfComponents != typename Array::IndexType(cellsCount) )
      throw std::length_error("Mismatched array size for <CellData> section: " + std::to_string(array.getSize())
                              + " (there are " + std::to_string(cellsCount) + " cells in the file)");
   openCellData();
   writeDataArray( array, name, numberOfComponents );
}

template< typename Mesh >
   template< typename Array >
void
VTUWriter< Mesh >::writeDataArray( const Array& array,
                                   const std::string& name,
                                   const int numberOfComponents )
{
   // use a host buffer if direct access to the array elements is not possible
   if( std::is_same< typename Array::DeviceType, Devices::Cuda >::value )
   {
      using HostArray = typename Array::template Self< std::remove_const_t< typename Array::ValueType >, Devices::Host, typename Array::IndexType >;
      HostArray hostBuffer;
      hostBuffer = array;
      writeDataArray( hostBuffer, name, numberOfComponents );
      return;
   }

   if( numberOfComponents != 0 && numberOfComponents != 1 && numberOfComponents != 3 )
      throw std::logic_error("Unsupported numberOfComponents parameter: " + std::to_string(numberOfComponents));

   // write DataArray header
   using ValueType = decltype(array[0]);
   str << "<DataArray type=\"" << VTK::getTypeName( ValueType{} ) << "\"";
   str << " Name=\"" << name << "\"";
   if( numberOfComponents > 0 )
      str << " NumberOfComponents=\"" << numberOfComponents << "\"";
   str << " format=\"" << ((format == VTK::FileFormat::ascii) ? "ascii" : "binary") << "\">\n";

   switch( format )
   {
      case VTK::FileFormat::ascii:
         str.precision( std::numeric_limits< typename Array::ValueType >::digits10 );
         for( typename Array::IndexType i = 0; i < array.getSize(); i++ )
            // If Array::ValueType is uint8_t, it might be a typedef for unsigned char, which
            // would be normally printed as char rather than a number. Hence, we use the trick
            // with unary operator+, see https://stackoverflow.com/a/28414758
            str << +array[i] << " ";
         str << "\n";
         break;
      case VTK::FileFormat::zlib_compressed:
#ifdef HAVE_ZLIB
         write_compressed_block< HeaderType >( array.getData(), array.getSize(), str );
         str << "\n";
         break;
#endif
         // fall through to binary if HAVE_ZLIB is not defined
      case VTK::FileFormat::binary:
         base64::write_encoded_block< HeaderType >( array.getData(), array.getSize(), str );
         str << "\n";
         break;
   }

   // write DataArray footer
   str << "</DataArray>\n";
}

template< typename Mesh >
void
VTUWriter< Mesh >::writePoints( const Mesh& mesh )
{
   // copy all coordinates into a contiguous array
   using BufferType = Containers::Array< typename Mesh::RealType, Devices::Host, typename Mesh::GlobalIndexType >;
   BufferType buffer( 3 * pointsCount );
   typename Mesh::GlobalIndexType k = 0;
   for( std::uint64_t i = 0; i < pointsCount; i++ ) {
      const auto& vertex = mesh.template getEntity< typename Mesh::Vertex >( i );
      const auto& point = vertex.getPoint();
      for( int j = 0; j < point.getSize(); j++ )
         buffer[ k++ ] = point[ j ];
      // VTK needs zeros for unused dimensions
      for( int j = point.getSize(); j < 3; j++ )
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
#ifdef HAVE_ZLIB
   if( format == VTK::FileFormat::zlib_compressed )
      str << " compressor=\"vtkZLibDataCompressor\"";
#endif
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
