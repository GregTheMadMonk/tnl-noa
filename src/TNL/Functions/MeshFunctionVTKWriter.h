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

template< typename MeshFunction,
          bool = std::is_fundamental< typename MeshFunction::RealType >::value >
class MeshFunctionVTKWriter
{
   using MeshType = typename MeshFunction::MeshType;
   using MeshWriter = Meshes::Writers::VTKWriter< MeshType >;
   using EntityType = typename MeshType::template EntityType< MeshFunction::getEntitiesDimension() >;
   using GlobalIndex = typename MeshType::GlobalIndexType;

public:
   static bool write( const MeshFunction& function,
                      std::ostream& str,
                      const String& functionName = "cellFunctionValues",
                      Meshes::Writers::VTKFileFormat format = Meshes::Writers::VTKFileFormat::ASCII )
   {
      const MeshType& mesh = function.getMesh();
      MeshWriter::template writeEntities< MeshFunction::getEntitiesDimension() >( mesh, str, format );
      appendFunction( function, str, format, functionName );
      return true;
   }

   // VTK supports writing multiple functions into the same file.
   // You can call this after 'write', which initializes the mesh entities,
   // with different function name.
   static void appendFunction( const MeshFunction& function,
                               std::ostream& str,
                               Meshes::Writers::VTKFileFormat format,
                               const String& functionName )
   {
      const MeshType& mesh = function.getMesh();
      const GlobalIndex entitiesCount = mesh.template getEntitiesCount< EntityType >();
      str << std::endl << "CELL_DATA " << entitiesCount << std::endl;
      str << "SCALARS " << functionName << " " << getType< typename MeshFunction::RealType >() << " 1" << std::endl;
      str << "LOOKUP_TABLE default" << std::endl;
      for( GlobalIndex i = 0; i < entitiesCount; i++ ) {
         const typename MeshFunction::RealType value = function.getData().getElement( i );
         using Meshes::Writers::__impl::writeReal;
         writeReal( format, str, value );
         if( format == Meshes::Writers::VTKFileFormat::ASCII )
            str << "\n";
      }
   }
};

template< typename MeshFunction >
class MeshFunctionVTKWriter< MeshFunction, false >
{
public:
   static bool write( const MeshFunction& function,
                      std::ostream& str,
                      const String& functionName = "cellFunctionValues",
                      Meshes::Writers::VTKFileFormat format = Meshes::Writers::VTKFileFormat::ASCII )
   {
      throw std::logic_error( "Unsupported RealType - VTKWriter supports only fundamental types." );
   }

   static void appendFunction( const MeshFunction& function,
                               std::ostream& str,
                               Meshes::Writers::VTKFileFormat format,
                               const String& functionName )
   {
      throw std::logic_error( "Unsupported RealType - VTKWriter supports only fundamental types." );
   }
};

} // namespace Functions
} // namespace TNL
