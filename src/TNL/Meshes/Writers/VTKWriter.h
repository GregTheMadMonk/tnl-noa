/***************************************************************************
                          VTKWriter.h  -  description
                             -------------------
    begin                : Mar 04, 2017
    copyright            : (C) 2017 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Meshes/Grid.h>
#include <TNL/Meshes/Mesh.h>
#include <TNL/Meshes/VTKTraits.h>

namespace TNL {
namespace Meshes {
namespace Writers {

namespace details {

template< typename Mesh, int EntityDimension > struct MeshEntitiesVTKWriter;
template< typename Mesh, int EntityDimension > struct MeshEntityTypesVTKWriter;

} // namespace details

template< typename Mesh >
class VTKWriter
{
   static_assert( Mesh::getMeshDimension() <= 3, "The VTK format supports only 1D, 2D and 3D meshes." );
   // TODO: check also world dimension when grids allow it
//   static_assert( Mesh::getWorldDimension() <= 3, "The VTK format supports only 1D, 2D and 3D meshes." );

   template< int EntityDimension >
   using EntitiesWriter = details::MeshEntitiesVTKWriter< Mesh, EntityDimension >;

   template< int EntityDimension >
   using EntityTypesWriter = details::MeshEntityTypesVTKWriter< Mesh, EntityDimension >;

public:
   using IndexType = typename Mesh::GlobalIndexType;

   VTKWriter() = delete;

   VTKWriter( std::ostream& str, VTK::FileFormat format = VTK::FileFormat::ascii )
   : str(str), format(format)
   {
      if( format != VTK::FileFormat::ascii && format != VTK::FileFormat::binary )
         throw std::domain_error("The Legacy VTK file formats support only ASCII and BINARY formats.");
   }

   template< int EntityDimension = Mesh::getMeshDimension() >
   void writeEntities( const Mesh& mesh );

   template< typename Array >
   void writeDataArray( const Array& array,
                        const String& name,
                        const int numberOfComponents = 1,
                        VTK::DataType dataType = VTK::DataType::CellData );

protected:
   void writeHeader( const Mesh& mesh );

   void writePoints( const Mesh& mesh );

   std::ostream& str;

   VTK::FileFormat format;

   // number of cells (in the VTK sense) written to the file
   IndexType cellsCount = 0;

   // number of points written to the file
   IndexType pointsCount = 0;

   // number of data arrays written in each section
   int cellDataArrays = 0;
   int pointDataArrays = 0;

   // indicator of the current section
   VTK::DataType currentSection = VTK::DataType::CellData;
};

} // namespace Writers
} // namespace Meshes
} // namespace TNL

#include <TNL/Meshes/Writers/VTKWriter.hpp>
