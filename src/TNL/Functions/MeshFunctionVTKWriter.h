/***************************************************************************
                          MeshFunctionVTKWriter.h  -  description
                             -------------------
    begin                : Jan 28, 2016
    copyright            : (C) 2016 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Meshes/Writers/VTKWriter.h>

namespace TNL {
namespace Functions {

template< typename MeshFunction >
class MeshFunctionVTKWriter
: protected Meshes::Writers::VTKWriter< typename MeshFunction::MeshType >
{
   using MeshType = typename MeshFunction::MeshType;
   using EntityType = typename MeshType::template EntityType< MeshFunction::getEntitiesDimension() >;
   using GlobalIndex = typename MeshType::GlobalIndexType;

public:
   MeshFunctionVTKWriter( std::ostream& str,
                          Meshes::VTK::FileFormat format = Meshes::VTK::FileFormat::ascii )
   : Meshes::Writers::VTKWriter< MeshType >( str, format )
   {}

   void write( const MeshFunction& function,
               const String& functionName = "cellFunctionValues" )
   {
      const MeshType& mesh = function.getMesh();
      this->template writeEntities< MeshFunction::getEntitiesDimension() >( mesh );
      appendFunction( function, functionName );
   }

   // VTK supports writing multiple functions into the same file.
   // You can call this after 'write', which initializes the mesh entities,
   // with different function name.
   void appendFunction( const MeshFunction& function,
                        const String& functionName )
   {
      if( MeshFunction::getEntitiesDimension() == 0 )
         this->writeDataArray( function.getData(), functionName, 1, Meshes::VTK::DataType::PointData );
      else
         this->writeDataArray( function.getData(), functionName, 1, Meshes::VTK::DataType::CellData );
   }
};

} // namespace Functions
} // namespace TNL
