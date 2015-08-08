/***************************************************************************
                          tnlMeshWriterVTKLegacy.h  -  description
                             -------------------
    begin                : Mar 20, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/


#ifndef TNLMESHWRITERVTKLEGACY_H_
#define TNLMESHWRITERVTKLEGACY_H_

#include <fstream>
#include <istream>
#include <sstream>
#include <iomanip>

using namespace std;

class tnlMeshWriterVTKLegacy
{
   public:

   template< typename MeshType >
   static bool write( const tnlString& fileName,
                          MeshType& mesh,
                          bool verbose )
   {
      if( MeshType::dimensions > 3 )
      {
         cerr << "You try to write mesh with " << MeshType::dimensions
              << "dimensions but VTK legacy format supports only 1D, 2D and 3D meshes." << endl;
         return false;
      }
      fstream outputFile;
      outputFile.open( fileName.getString(), ios::out );
      if( ! outputFile )
      {
         cerr << "I am not able to open the output file " << fileName << "." << endl;
         return false;
      }
      outputFile << setprecision( 6 );
      outputFile << fixed;

      if( ! writeMesh( outputFile, mesh, verbose ) )
         return false;
   }

   template< typename MeshType >
   static bool writeMesh( ostream& file,
                          MeshType& mesh,
                          bool verbose )
   {
      file << "# vtk DataFile Version 2.0" << endl;
      file << "TNL Mesh" << endl;
      file << "ASCII" << endl;
      file << "DATASET UNSTRUCTURED_GRID" << endl;
      file << endl;
      file << "POINTS " << mesh.template getNumberOfEntities< 0 >() << " float" << endl;
      for( int i = 0; i < mesh.template getNumberOfEntities< 0 >(); i++ )
      {
         file << mesh.template getEntity< 0 >( i ).getPoint();
      }
   }


};



#endif /* TNLMESHWRITERVTKLEGACY_H_ */
