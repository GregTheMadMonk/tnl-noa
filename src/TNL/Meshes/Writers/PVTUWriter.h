/***************************************************************************
                          PVTUWriter.h  -  description
                             -------------------
    begin                : Apr 17, 2020
    copyright            : (C) 2020 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovský

#pragma once

#include <TNL/Meshes/Writers/VTUWriter.h>
#include <TNL/Meshes/DistributedMeshes/DistributedMesh.h>

namespace TNL {
namespace Meshes {
namespace Writers {

// NOTE: Mesh should be the local mesh type, not DistributedMesh
template< typename Mesh >
class PVTUWriter
{
   using HeaderType = std::uint64_t;
public:
   using MeshRealType = typename Mesh::RealType;
   using IndexType = typename Mesh::GlobalIndexType;

   PVTUWriter() = delete;

   PVTUWriter( std::ostream& str, VTK::FileFormat format = VTK::FileFormat::zlib_compressed )
   : str(str.rdbuf()), format(format)
   {}

   // If desired, cycle and time of the simulation can put into the file. This follows the instructions at
   // http://www.visitusers.org/index.php?title=Time_and_Cycle_in_VTK_files
   void writeMetadata( std::int32_t cycle = -1, double time = -1 );

   template< int EntityDimension = Mesh::getMeshDimension() >
   void writeEntities( const DistributedMeshes::DistributedMesh< Mesh >& distributedMesh );

   template< int EntityDimension = Mesh::getMeshDimension() >
   void writeEntities( const Mesh& mesh,
                       const unsigned GhostLevel = 0,
                       const unsigned MinCommonVertices = 0 );

   template< typename ValueType >
   void writePPointData( const String& name,
                         const int numberOfComponents = 1 );

   template< typename ValueType >
   void writePCellData( const String& name,
                        const int numberOfComponents = 1 );

   template< typename ValueType >
   void writePDataArray( const String& name,
                         const int numberOfComponents = 1 );

   // add a single piece and return its source path
   // (useful for sequential writing, e.g. from tnl-decompose-mesh)
   std::string addPiece( const String& mainFileName,
                         const unsigned subdomainIndex );

   // add all pieces and return the source path for the current rank
   // (useful for parallel writing)
   std::string addPiece( const String& mainFileName,
                         const MPI_Comm communicator );

   ~PVTUWriter();

protected:
   void writeHeader( const unsigned GhostLevel = 0,
                     const unsigned MinCommonVertices = 0 );

   void writePoints( const Mesh& mesh );

   void writeFooter();

   std::ostream str;

   VTK::FileFormat format;

   // indicator if the <VTKFile> tag is open
   bool vtkfileOpen = false;

   // indicators if a <PCellData> tag is open or closed
   bool pCellDataOpen = false;
   bool pCellDataClosed = false;

   // indicators if a <PPointData> tag is open or closed
   bool pPointDataOpen = false;
   bool pPointDataClosed = false;

   void openPCellData();
   void closePCellData();
   void openPPointData();
   void closePPointData();
};

} // namespace Writers
} // namespace Meshes
} // namespace TNL

#include <TNL/Meshes/Writers/PVTUWriter.hpp>
