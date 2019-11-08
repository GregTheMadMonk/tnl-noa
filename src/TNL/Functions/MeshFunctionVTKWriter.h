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
{
   using MeshType = typename MeshFunction::MeshType;
   using MeshWriter = Meshes::Writers::VTKWriter< MeshType >;
   using EntityType = typename MeshType::template EntityType< MeshFunction::getEntitiesDimension() >;
   using GlobalIndex = typename MeshType::GlobalIndexType;

public:
   static bool write( const MeshFunction& function,
                      std::ostream& str,
                      const double& scale = 1.0,
                      const String& functionName = "cellFunctionValues" )
   {
      const MeshType& mesh = function.getMesh();
      MeshWriter::template writeEntities< MeshFunction::getEntitiesDimension() >( mesh, str );
      appendFunction( function, str, functionName, scale );
      return true;
   }

   // VTK supports writing multiple functions into the same file.
   // You can call this after 'write', which initializes the mesh entities,
   // with different function name.
   static void appendFunction( const MeshFunction& function,
                               std::ostream& str,
                               const String& functionName,
                               const double& scale = 1.0 )
   {
      const MeshType& mesh = function.getMesh();
      const GlobalIndex entitiesCount = mesh.template getEntitiesCount< EntityType >();
      str << std::endl << "CELL_DATA " << entitiesCount << std::endl;
      str << "SCALARS " << functionName << " " << getType< typename MeshFunction::RealType >() << " 1" << std::endl;
      str << "LOOKUP_TABLE default" << std::endl;
      for( GlobalIndex i = 0; i < entitiesCount; i++ ) {
         str << scale * function.getData().getElement( i ) << "\n";
      }
   }
};

} // namespace Functions
} // namespace TNL
