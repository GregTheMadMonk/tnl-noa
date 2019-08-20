/***************************************************************************
                          VectorFieldVTKWriter.h  -  description
                             -------------------
    begin                : Jan 10, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Meshes/Writers/VTKWriter.h>

namespace TNL {
namespace Functions {

template< typename VectorField >
class VectorFieldVTKWriter
{
   using MeshType = typename VectorField::MeshType;
   using MeshWriter = Meshes::Writers::VTKWriter< MeshType >;
   using EntityType = typename MeshType::template EntityType< VectorField::getEntitiesDimension() >;
   using GlobalIndex = typename MeshType::GlobalIndexType;

public:
   static bool write( const VectorField& field,
                      std::ostream& str,
                      const double& scale = 1.0,
                      const String& fieldName = "cellVectorFieldValues" )
   {
      const MeshType& mesh = field.getMesh();
      MeshWriter::template writeEntities< VectorField::getEntitiesDimension() >( mesh, str );
      appendField( field, str, fieldName, scale );
      return true;
   }

   // VTK supports writing multiple fields into the same file.
   // You can call this after 'write', which initializes the mesh entities,
   // with different field name.
   static void appendField( const VectorField& field,
                            std::ostream& str,
                            const String& fieldName,
                            const double& scale = 1.0 )
   {
      const MeshType& mesh = field.getMesh();
      const GlobalIndex entitiesCount = mesh.template getEntitiesCount< EntityType >();
      str << std::endl << "CELL_DATA " << entitiesCount << std::endl;
      str << "VECTORS " << fieldName << " " << getType< typename VectorField::RealType >() << " 1" << std::endl;
      for( GlobalIndex i = 0; i < entitiesCount; i++ ) {
         const typename VectorField::VectorType vector = field.getElement( i );
         static_assert( VectorField::getVectorDimension() <= 3, "The VTK format supports only up to 3D vector fields." );
         for( int i = 0; i < 3; i++ )
            str << scale * ( i < vector.getSize() ? vector[ i ] : 0.0 ) << " ";
         str << "\n";
      }
   }
};

} // namespace Functions
} // namespace TNL
