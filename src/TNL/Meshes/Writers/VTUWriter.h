/***************************************************************************
                          VTUWriter.h  -  description
                             -------------------
    begin                : Mar 18, 2020
    copyright            : (C) 2020 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovský

#pragma once

#include <TNL/Meshes/Grid.h>
#include <TNL/Meshes/Mesh.h>
#include <TNL/Meshes/VTKTraits.h>

namespace TNL {
namespace Meshes {
namespace Writers {

namespace details {

template< typename Mesh, int EntityDimension > struct MeshEntitiesVTUCollector;

} // namespace details

template< typename Mesh >
class VTUWriter
{
   static_assert( Mesh::getMeshDimension() <= 3, "The VTK format supports only 1D, 2D and 3D meshes." );
   // TODO: check also world dimension when grids allow it
//   static_assert( Mesh::getWorldDimension() <= 3, "The VTK format supports only 1D, 2D and 3D meshes." );

   template< int EntityDimension >
   using EntitiesCollector = details::MeshEntitiesVTUCollector< Mesh, EntityDimension >;

   using HeaderType = std::uint64_t;
public:
   using MeshRealType = typename Mesh::RealType;
   using IndexType = typename Mesh::GlobalIndexType;

   VTUWriter() = delete;

   VTUWriter( std::ostream& str, VTK::FileFormat format = VTK::FileFormat::ascii )
   : str(str), format(format)
   {}

   // If desired, cycle and time of the simulation can put into the file. This follows the instructions at
   // http://www.visitusers.org/index.php?title=Time_and_Cycle_in_VTK_files
   void writeMetadata( std::int32_t cycle = -1, double time = -1 );

   template< int EntityDimension = Mesh::getMeshDimension() >
   void writeEntities( const Mesh& mesh );

   template< typename Array >
   void writeDataArray( const Array& array,
                        const String& name,
                        const int numberOfComponents = 1 );

   ~VTUWriter();

protected:
   void writeHeader();

   void writeFooter();

   void writePoints( const Mesh& mesh );

   std::ostream& str;

   VTK::FileFormat format;

   // number of points written to the file
   IndexType pointsCount = 0;

   // number of cells (in the VTK sense) written to the file
   IndexType cellsCount = 0;

   // indicator if the <VTKFile> tag is open
   bool vtkfileOpen = false;

   // indicator if a <Piece> tag is open
   bool pieceOpen = false;

   // number of data arrays written in each section
   int cellDataArrays = 0;
   int pointDataArrays = 0;

   // indicator of the current section
   VTK::DataType currentSection = VTK::DataType::CellData;
};

} // namespace Writers
} // namespace Meshes
} // namespace TNL

#include <TNL/Meshes/Writers/VTUWriter.hpp>
