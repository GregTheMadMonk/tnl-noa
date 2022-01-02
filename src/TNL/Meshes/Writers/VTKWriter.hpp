/***************************************************************************
                          VTKWriter.hpp  -  description
                             -------------------
    begin                : Mar 04, 2017
    copyright            : (C) 2017 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <limits>

#include <TNL/Meshes/Writers/VTKWriter.h>
#include <TNL/Meshes/Writers/VerticesPerEntity.h>
#include <TNL/Meshes/Writers/EntitiesListSize.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/Endianness.h>

namespace TNL {
namespace Meshes {
namespace Writers {

namespace details {

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
struct MeshEntitiesVTKWriter
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
struct MeshEntitiesVTKWriter< Mesh, 0 >
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
struct MeshEntitiesVTKWriter< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, 1 >
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
struct MeshEntitiesVTKWriter< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, 0 >
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
struct MeshEntitiesVTKWriter< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, 2 >
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
struct MeshEntitiesVTKWriter< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, 1 >
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
struct MeshEntitiesVTKWriter< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, 0 >
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
struct MeshEntitiesVTKWriter< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, 3 >
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
struct MeshEntitiesVTKWriter< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, 2 >
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
struct MeshEntitiesVTKWriter< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, 1 >
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
struct MeshEntitiesVTKWriter< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, 0 >
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
struct MeshEntityTypesVTKWriter
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
struct MeshEntityTypesVTKWriter< Grid< Dimension, MeshReal, Device, MeshIndex >, EntityDimension >
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

} // namespace details

template< typename Mesh >
void
VTKWriter< Mesh >::writeMetadata( int cycle, double time )
{
   if( ! headerWritten )
      writeHeader();

   int n_metadata = 0;
   if( cycle >= 0 )
      ++n_metadata;
   if( time >= 0 )
      ++n_metadata;
   if( n_metadata > 0 )
      str << "FIELD FieldData " << n_metadata << "\n";
   if( cycle >= 0 ) {
      str << "CYCLE 1 1 int\n";
      details::writeInt( format, str, cycle );
      str << "\n";
   }
   if( time >= 0 ) {
      str << "TIME 1 1 double\n";
      details::writeReal( format, str, time );
      str << "\n";
   }
}

template< typename Mesh >
   template< int EntityDimension >
void
VTKWriter< Mesh >::writeEntities( const Mesh& mesh )
{
   if( ! headerWritten )
      writeHeader();
   writePoints( mesh );

   using EntityType = typename Mesh::template EntityType< EntityDimension >;
   cellsCount = mesh.template getEntitiesCount< EntityType >();
   const std::uint64_t cellsListSize = EntitiesListSize< Mesh, EntityDimension >::getSize( mesh );

   str << std::endl << "CELLS " << cellsCount << " " << cellsListSize << std::endl;
   EntitiesWriter< EntityDimension >::exec( mesh, str, format );

   str << std::endl << "CELL_TYPES " << cellsCount << std::endl;
   EntityTypesWriter< EntityDimension >::exec( mesh, str, format );
}

template< typename Mesh >
   template< typename Array >
void
VTKWriter< Mesh >::writePointData( const Array& array,
                                   const std::string& name,
                                   const int numberOfComponents )
{
   if( array.getSize() / numberOfComponents != typename Array::IndexType(pointsCount) )
      throw std::length_error("Mismatched array size for POINT_DATA section: " + std::to_string(array.getSize())
                              + " (there are " + std::to_string(pointsCount) + " points in the file)");

   // check that we won't start the section second time
   if( currentSection != VTK::DataType::PointData && cellDataArrays * pointDataArrays != 0 )
      throw std::logic_error("The requested data section is not the current section and it has already been written.");
   currentSection = VTK::DataType::PointData;

   // start the appropriate section if necessary
   if( pointDataArrays == 0 )
      str << std::endl << "POINT_DATA " << pointsCount << std::endl;
   ++pointDataArrays;

   writeDataArray( array, name, numberOfComponents );
}

template< typename Mesh >
   template< typename Array >
void
VTKWriter< Mesh >::writeCellData( const Array& array,
                                  const std::string& name,
                                  const int numberOfComponents )
{
   if( array.getSize() / numberOfComponents != typename Array::IndexType(cellsCount) )
      throw std::length_error("Mismatched array size for CELL_DATA section: " + std::to_string(array.getSize())
                              + " (there are " + std::to_string(cellsCount) + " cells in the file)");

   // check that we won't start the section second time
   if( currentSection != VTK::DataType::CellData && cellDataArrays * pointDataArrays != 0 )
      throw std::logic_error("The requested data section is not the current section and it has already been written.");
   currentSection = VTK::DataType::CellData;

   // start the appropriate section if necessary
   if( cellDataArrays == 0 )
      str << std::endl << "CELL_DATA " << cellsCount << std::endl;
   ++cellDataArrays;

   writeDataArray( array, name, numberOfComponents );
}

template< typename Mesh >
   template< typename Array >
void
VTKWriter< Mesh >::writeDataArray( const Array& array,
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

   if( numberOfComponents != 1 && numberOfComponents != 3 )
      throw std::logic_error("Unsupported numberOfComponents parameter: " + std::to_string(numberOfComponents));

   // write DataArray header
   if( numberOfComponents == 1 ) {
      str << "SCALARS " << name << " " << getType< typename Array::ValueType >() << " 1" << std::endl;
      str << "LOOKUP_TABLE default" << std::endl;
   }
   else {
      str << "VECTORS " << name << " " << getType< typename Array::ValueType >() << " 1" << std::endl;
   }

   using Meshes::Writers::details::writeReal;
   for( typename Array::IndexType i = 0; i < array.getSize(); i++ ) {
      writeReal( format, str, array[i] );
      if( format == VTK::FileFormat::ascii )
         str << "\n";
   }
}

template< typename Mesh >
void
VTKWriter< Mesh >::writePoints( const Mesh& mesh )
{
   using details::writeReal;
   pointsCount = mesh.template getEntitiesCount< typename Mesh::Vertex >();
   str << "POINTS " << pointsCount << " " << getType< typename Mesh::RealType >() << std::endl;
   for( std::uint64_t i = 0; i < pointsCount; i++ ) {
      const auto& vertex = mesh.template getEntity< typename Mesh::Vertex >( i );
      const auto& point = vertex.getPoint();
      for( int j = 0; j < point.getSize(); j++ )
         writeReal( format, str, point[ j ] );
      // VTK needs zeros for unused dimensions
      for( int j = point.getSize(); j < 3; j++ )
         writeReal( format, str, (typename Mesh::PointType::RealType) 0 );
      if( format == VTK::FileFormat::ascii )
         str << "\n";
   }
}

template< typename Mesh >
void
VTKWriter< Mesh >::writeHeader()
{
    str << "# vtk DataFile Version 2.0\n"
        << "TNL DATA\n"
        << ((format == VTK::FileFormat::ascii) ? "ASCII\n" : "BINARY\n")
        << "DATASET UNSTRUCTURED_GRID\n";
    headerWritten = true;
}

} // namespace Writers
} // namespace Meshes
} // namespace TNL
