/***************************************************************************
                          VTKWriter.h  -  description
                             -------------------
    begin                : Mar 04, 2017
    copyright            : (C) 2017 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <limits>

#include <TNL/Meshes/Grid.h>
#include <TNL/Meshes/Mesh.h>

namespace TNL {
namespace Meshes {
namespace Writers {

enum class VTKFileFormat
{
   ASCII,
   BINARY
};

enum class VTKDataType
{
   CellData,
   PointData
};

namespace __impl {

template< typename Mesh, int EntityDimension > struct MeshEntitiesVTKWriter;
template< typename Mesh, int EntityDimension > struct MeshEntityTypesVTKWriter;

} // namespace __impl

template< typename Mesh >
class VTKWriter
{
   static_assert( Mesh::getMeshDimension() <= 3, "The VTK format supports only 1D, 2D and 3D meshes." );
   // TODO: check also world dimension when grids allow it
//   static_assert( Mesh::getWorldDimension() <= 3, "The VTK format supports only 1D, 2D and 3D meshes." );

   template< int EntityDimension >
   using EntitiesWriter = __impl::MeshEntitiesVTKWriter< Mesh, EntityDimension >;

   template< int EntityDimension >
   using EntityTypesWriter = __impl::MeshEntityTypesVTKWriter< Mesh, EntityDimension >;

public:
   using IndexType = typename Mesh::GlobalIndexType;

   VTKWriter() = delete;

   VTKWriter( std::ostream& str, VTKFileFormat format = VTKFileFormat::ASCII )
   : str(str), format(format)
   {}

   void writeAllEntities( const Mesh& mesh );

   template< int EntityDimension = Mesh::getMeshDimension() >
   void writeEntities( const Mesh& mesh );

   template< typename Array >
   void writeDataArray( const Array& array,
                        const String& name,
                        const int numberOfComponents = 1,
                        VTKDataType dataType = VTKDataType::CellData );

protected:
   void writeHeader( const Mesh& mesh );

   void writePoints( const Mesh& mesh );

   std::ostream& str;

   VTKFileFormat format;

   // number of cells (in the VTK sense) written to the file
   IndexType cellsCount = 0;

   // number of points written to the file
   IndexType pointsCount = 0;

   // number of data arrays written in each section
   int cellDataArrays = 0;
   int pointDataArrays = 0;

   // indicator of the current section
   VTKDataType currentSection = VTKDataType::CellData;
};

} // namespace Writers
} // namespace Meshes
} // namespace TNL

#include <TNL/Meshes/Writers/VTKWriter.hpp>
