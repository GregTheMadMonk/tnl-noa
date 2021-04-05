/***************************************************************************
                          VTUReader.h  -  description
                             -------------------
    begin                : Mar 21, 2020
    copyright            : (C) 2020 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovský

#pragma once

#include <TNL/Meshes/Readers/XMLVTK.h>

namespace TNL {
namespace Meshes {
namespace Readers {

class VTUReader
: public XMLVTK
{
#ifdef HAVE_TINYXML2
   void readUnstructuredGrid()
   {
      using namespace tinyxml2;
      const XMLElement* piece = getChildSafe( datasetElement, "Piece" );
      if( piece->NextSiblingElement( "Piece" ) )
         // ambiguity - throw error, we don't know which piece to parse (or all of them?)
         throw MeshReaderError( "VTUReader", "the serial UnstructuredGrid file contains more than one <Piece> element" );
      NumberOfPoints = getAttributeInteger( piece, "NumberOfPoints" );
      NumberOfCells = getAttributeInteger( piece, "NumberOfCells" );

      // verify points
      const XMLElement* points = getChildSafe( piece, "Points" );
      const XMLElement* pointsData = verifyHasOnlyOneChild( points, "DataArray" );
      verifyDataArray( pointsData );
      const std::string pointsDataName = getAttributeString( pointsData, "Name" );
      if( pointsDataName != "Points" )
         throw MeshReaderError( "VTUReader", "the <Points> tag does not contain a <DataArray> with Name=\"Points\" attribute" );

      // verify cells
      const XMLElement* cells = getChildSafe( piece, "Cells" );
      const XMLElement* connectivity = getDataArrayByName( cells, "connectivity" );
      const XMLElement* offsets = getDataArrayByName( cells, "offsets" );
      const XMLElement* types = getDataArrayByName( cells, "types" );

      // read the points, connectivity, offsets and types into intermediate arrays
      pointsArray = readDataArray( pointsData, "Points" );
      pointsType = VTKDataTypes.at( getAttributeString( pointsData, "type" ) );
      connectivityArray = readDataArray( connectivity, "connectivity" );
      connectivityType = VTKDataTypes.at( getAttributeString( connectivity, "type" ) );
      offsetsArray = readDataArray( offsets, "offsets" );
      offsetsType = VTKDataTypes.at( getAttributeString( offsets, "type" ) );
      typesArray = readDataArray( types, "types" );
      typesType = VTKDataTypes.at( getAttributeString( types, "type" ) );

      // connectivity and offsets must have the same type
      if( connectivityType != offsetsType )
         throw MeshReaderError( "VTUReader", "the \"connectivity\" and \"offsets\" array do not have the same type ("
                            + connectivityType + " vs " + offsetsType + ")" );
      // cell types can be only uint8_t
      if( typesType != "std::uint8_t" )
         throw MeshReaderError( "VTUReader", "unsupported data type for the Name=\"types\" array" );

      using mpark::visit;
      // validate points
      visit( [this](auto&& array) {
               // check array size
               if( array.size() != 3 * NumberOfPoints )
                  throw MeshReaderError( "VTUReader", "invalid size of the Points data array (" + std::to_string(array.size())
                                                      + " vs " + std::to_string(NumberOfPoints) + ")" );
               // set spaceDimension
               spaceDimension = 1;
               std::size_t i = 0;
               for( auto c : array ) {
                  if( c != 0 ) {
                     int dim = i % 3 + 1;
                     spaceDimension = std::max( spaceDimension, dim );
                  }
                  ++i;
               }
            },
            pointsArray
         );
      // validate types
      visit( [this](auto&& array) {
               // check array size
               if( array.size() != NumberOfCells )
                  throw MeshReaderError( "VTUReader", "size of the types data array does not match the NumberOfCells attribute" );
               // check empty mesh
               if( array.size() == 0 )
                  return;
               cellShape = (VTK::EntityShape) array[0];
               meshDimension = getEntityDimension( cellShape );
               using PolygonShapeGroupChecker = VTK::EntityShapeGroupChecker< VTK::EntityShape::Polygon >;
               //TODO: uncomment line below later for polyhedrals
               //using PolyhedralShapeGroupChecker = VTK::EntityShapeGroupChecker< VTK::EntityShape::Polyhedral >;

               // TODO: check only entities of the same dimension (edges, faces and cells separately)
               for( auto c : array )
               {
                  VTK::EntityShape entityShape = (VTK::EntityShape) c;
                  if( entityShape != cellShape )
                  {
                     if( PolygonShapeGroupChecker::bothBelong( cellShape, entityShape ) )
                     {
                        cellShape = PolygonShapeGroupChecker::GeneralShape;
                     }
                     //TODO: add group check for polyhedrals later
                     /*else if( PolyhedralEntityShapeGroupChecker::bothBelong( cellShape, entityShape ) )
                     {
                        cellShape = PolyhedralEntityShapeGroupChecker::GeneralShape;
                     }*/
                     else
                     {
                        throw MeshReaderError( "VTUReader", "Mixed unstructured meshes are not supported. There are cells with type "
                                                         + VTK::getShapeName(cellShape) + " and " + VTK::getShapeName(entityShape) + "." );
                     }
                  }
               }
            },
            typesArray
         );
      // validate offsets
      std::size_t max_offset = 0;
      visit( [this, &max_offset](auto&& array) mutable {
               if( array.size() != NumberOfCells )
                  throw MeshReaderError( "VTUReader", "size of the offsets data array does not match the NumberOfCells attribute" );
               for( auto c : array ) {
                  if( c <= (decltype(c)) max_offset )
                     throw MeshReaderError( "VTUReader", "the offsets array is not monotonically increasing" );
                  max_offset = c;
               }
            },
            offsetsArray
         );
      // validate connectivity
      visit( [this, max_offset](auto&& array) {
               if( array.size() != max_offset )
                  throw MeshReaderError( "VTUReader", "size of the connectivity data array does not match the offsets array" );
               for( auto c : array ) {
                  if( c < 0 || (std::size_t) c >= NumberOfPoints )
                     throw MeshReaderError( "VTUReader", "connectivity index " + std::to_string(c) + " is out of range" );
               }
            },
            connectivityArray
         );
   }
#endif

public:
   VTUReader() = default;

   VTUReader( const std::string& fileName )
   : XMLVTK( fileName )
   {}

   virtual void detectMesh() override
   {
#ifdef HAVE_TINYXML2
      reset();
      try {
         openVTKFile();
      }
      catch( const MeshReaderError& ) {
         reset();
         throw;
      }

      // verify file type
      if( fileType == "UnstructuredGrid" )
         readUnstructuredGrid();
      else
         throw MeshReaderError( "VTUReader", "the reader cannot read data of the type " + fileType + ". Use a different reader if possible." );

      // indicate success by setting the mesh type
      meshType = "Meshes::Mesh";
#else
      throw_no_tinyxml();
#endif
   }
};

} // namespace Readers
} // namespace Meshes
} // namespace TNL
