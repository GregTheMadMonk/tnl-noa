/***************************************************************************
                          VTKWriter.h  -  description
                             -------------------
    begin                : Mar 04, 2017
    copyright            : (C) 2017 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Meshes/VTKTraits.h>

namespace TNL {
namespace Meshes {
//! \brief Namespace for mesh writers.
namespace Writers {

namespace details {

template< typename Mesh, int EntityDimension > struct MeshEntitiesVTKWriter;
template< typename Mesh, int EntityDimension > struct MeshEntityTypesVTKWriter;

} // namespace details

template< typename Mesh >
class VTKWriter
{
   static_assert( Mesh::getMeshDimension() <= 3, "The VTK format supports only 1D, 2D and 3D meshes." );
   // TODO: check also space dimension when grids allow it
//   static_assert( Mesh::getSpaceDimension() <= 3, "The VTK format supports only 1D, 2D and 3D meshes." );

   template< int EntityDimension >
   using EntitiesWriter = details::MeshEntitiesVTKWriter< Mesh, EntityDimension >;

   template< int EntityDimension >
   using EntityTypesWriter = details::MeshEntityTypesVTKWriter< Mesh, EntityDimension >;

public:

   VTKWriter() = delete;

   VTKWriter( std::ostream& str, VTK::FileFormat format = VTK::FileFormat::binary )
   : str(str.rdbuf()), format(format)
   {
      if( format != VTK::FileFormat::ascii && format != VTK::FileFormat::binary )
         throw std::domain_error("The Legacy VTK file formats support only ASCII and BINARY formats.");
   }

   // If desired, cycle and time of the simulation can put into the file. This follows the instructions at
   // http://www.visitusers.org/index.php?title=Time_and_Cycle_in_VTK_files
   void writeMetadata( std::int32_t cycle = -1, double time = -1 );

   template< int EntityDimension = Mesh::getMeshDimension() >
   void writeEntities( const Mesh& mesh );

   template< typename Array >
   void writePointData( const Array& array,
                        const String& name,
                        const int numberOfComponents = 1 );

   template< typename Array >
   void writeCellData( const Array& array,
                       const String& name,
                       const int numberOfComponents = 1 );

   template< typename Array >
   void writeDataArray( const Array& array,
                        const String& name,
                        const int numberOfComponents = 1 );

protected:
   void writePoints( const Mesh& mesh );

   void writeHeader();

   std::ostream str;

   VTK::FileFormat format;

   // number of cells (in the VTK sense) written to the file
   std::uint64_t cellsCount = 0;

   // number of points written to the file
   std::uint64_t pointsCount = 0;

   // indicator if the header has been written
   bool headerWritten = false;

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
