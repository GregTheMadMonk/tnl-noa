/***************************************************************************
                          getMeshReader.h  -  description
                             -------------------
    begin                : Nov 7, 2020
    copyright            : (C) 2020 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovský

#pragma once

#include <memory>
#include <experimental/filesystem>

#include <TNL/Meshes/Readers/NetgenReader.h>
#include <TNL/Meshes/Readers/VTKReader.h>
#include <TNL/Meshes/Readers/VTUReader.h>
#include <TNL/Meshes/Readers/VTIReader.h>
#include <TNL/Meshes/Readers/PVTUReader.h>

namespace TNL {
namespace Meshes {
namespace Readers {

inline std::shared_ptr< MeshReader >
getMeshReader( const std::string& fileName,
               const std::string& fileFormat )
{
   namespace fs = std::experimental::filesystem;
   std::string format = fileFormat;
   if( format == "auto" ) {
      format = fs::path(fileName).extension();
      if( format.length() > 0 )
         // remove dot from the extension
         format = format.substr(1);
   }

   if( format == "ng" )
      return std::make_shared< Readers::NetgenReader >( fileName );
   else if( format == "vtk" )
      return std::make_shared< Readers::VTKReader >( fileName );
   else if( format == "vtu" )
      return std::make_shared< Readers::VTUReader >( fileName );
   else if( format == "vti" )
      return std::make_shared< Readers::VTIReader >( fileName );
   else if( format == "pvtu" )
      return std::make_shared< Readers::PVTUReader >( fileName );

   if( fileFormat == "auto" )
      std::cerr << "File '" << fileName << "' has unsupported format (based on the file extension): " << format << ".";
   else
      std::cerr << "Unsupported fileFormat parameter: " << fileFormat << ".";
   std::cerr << " Supported formats are 'ng', 'vtk', 'vtu', 'vti' and 'pvtu'." << std::endl;
   return nullptr;
}

} // namespace Readers
} // namespace Meshes
} // namespace TNL
