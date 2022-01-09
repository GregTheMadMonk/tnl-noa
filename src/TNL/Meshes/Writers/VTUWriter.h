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

#include <ostream>

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
   // TODO: check also space dimension when grids allow it
//   static_assert( Mesh::getSpaceDimension() <= 3, "The VTK format supports only 1D, 2D and 3D meshes." );

   template< int EntityDimension >
   using EntitiesCollector = details::MeshEntitiesVTUCollector< Mesh, EntityDimension >;

   using HeaderType = std::uint64_t;
public:

   VTUWriter() = delete;

   VTUWriter( std::ostream& str, VTK::FileFormat format = VTK::FileFormat::zlib_compressed )
   : str(str.rdbuf()), format(format)
   {}

   // If desired, cycle and time of the simulation can put into the file. This follows the instructions at
   // http://www.visitusers.org/index.php?title=Time_and_Cycle_in_VTK_files
   void writeMetadata( std::int32_t cycle = -1, double time = -1 );

   template< int EntityDimension = Mesh::getMeshDimension() >
   void writeEntities( const Mesh& mesh );

   template< typename Array >
   void writePointData( const Array& array,
                        const std::string& name,
                        const int numberOfComponents = 1 );

   template< typename Array >
   void writeCellData( const Array& array,
                       const std::string& name,
                       const int numberOfComponents = 1 );

   template< typename Array >
   void writeDataArray( const Array& array,
                        const std::string& name,
                        const int numberOfComponents = 1 );

   ~VTUWriter();

protected:
   void writePoints( const Mesh& mesh );

   void writeHeader();

   void writeFooter();

   std::ostream str;

   VTK::FileFormat format;

   // number of points written to the file
   std::uint64_t pointsCount = 0;

   // number of cells (in the VTK sense) written to the file
   std::uint64_t cellsCount = 0;

   // indicator if the <VTKFile> tag is open
   bool vtkfileOpen = false;

   // indicator if a <Piece> tag is open
   bool pieceOpen = false;

   // indicators if a <CellData> tag is open or closed
   bool cellDataOpen = false;
   bool cellDataClosed = false;

   // indicators if a <PointData> tag is open or closed
   bool pointDataOpen = false;
   bool pointDataClosed = false;

   void openCellData();
   void closeCellData();
   void openPointData();
   void closePointData();

   void closePiece();
};

} // namespace Writers
} // namespace Meshes
} // namespace TNL

#include <TNL/Meshes/Writers/VTUWriter.hpp>
