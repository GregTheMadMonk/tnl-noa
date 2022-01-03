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
#include <TNL/Meshes/Writers/detail/VTKEntitiesListSize.h>
#include <TNL/Meshes/Writers/detail/VTKMeshEntitiesWriter.h>
#include <TNL/Meshes/Grid.h>

namespace TNL {
namespace Meshes {
namespace Writers {

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
      detail::writeInt( format, str, cycle );
      str << "\n";
   }
   if( time >= 0 ) {
      str << "TIME 1 1 double\n";
      detail::writeReal( format, str, time );
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
   const std::uint64_t cellsListSize = detail::VTKEntitiesListSize< Mesh, EntityDimension >::getSize( mesh );

   str << std::endl << "CELLS " << cellsCount << " " << cellsListSize << std::endl;
   detail::VTKMeshEntitiesWriter< Mesh, EntityDimension >::exec( mesh, str, format );

   str << std::endl << "CELL_TYPES " << cellsCount << std::endl;
   detail::VTKMeshEntityTypesWriter< Mesh, EntityDimension >::exec( mesh, str, format );
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

   using detail::writeReal;
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
   using detail::writeReal;
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
