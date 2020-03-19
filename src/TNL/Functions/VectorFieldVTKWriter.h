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
: protected Meshes::Writers::VTKWriter< typename VectorField::MeshType >
{
   using MeshType = typename VectorField::MeshType;
   using MeshWriter = Meshes::Writers::VTKWriter< MeshType >;
   using EntityType = typename MeshType::template EntityType< VectorField::getEntitiesDimension() >;
   using GlobalIndex = typename MeshType::GlobalIndexType;

public:
   VectorFieldVTKWriter( std::ostream& str,
                         Meshes::VTK::FileFormat format = Meshes::VTK::FileFormat::ascii )
   : Meshes::Writers::VTKWriter< MeshType >( str, format )
   {}

   void write( const VectorField& field,
               const String& fieldName = "cellVectorFieldValues" )
   {
      const MeshType& mesh = field.getMesh();
      this->template writeEntities< VectorField::getEntitiesDimension() >( mesh );
      appendField( field, fieldName );
   }

   // VTK supports writing multiple fields into the same file.
   // You can call this after 'write', which initializes the mesh entities,
   // with different field name.
   void appendField( const VectorField& field,
                     const String& fieldName )
   {
      const MeshType& mesh = field.getMesh();
      const GlobalIndex entitiesCount = mesh.template getEntitiesCount< EntityType >();

      // copy all values from the vector field into a contiguous array
      using BufferType = Containers::Array< typename VectorField::RealType, Devices::Host, GlobalIndex >;
      BufferType buffer( 3 * entitiesCount );
      GlobalIndex j = 0;
      for( GlobalIndex i = 0; i < entitiesCount; i++ ) {
         const typename VectorField::VectorType vector = field.getElement( i );
         static_assert( VectorField::getVectorDimension() <= 3, "The VTK format supports only up to 3D vector fields." );
         for( int i = 0; i < 3; i++ )
            buffer[ j++ ] = ( i < vector.getSize() ? vector[ i ] : 0 );
      }

      // write the buffer
      if( VectorField::getEntitiesDimension() == 0 )
         this->writeDataArray( buffer, fieldName, 3, Meshes::VTK::DataType::PointData );
      else
         this->writeDataArray( buffer, fieldName, 3, Meshes::VTK::DataType::CellData );
   }
};

} // namespace Functions
} // namespace TNL
