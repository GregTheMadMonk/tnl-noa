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
   static bool writeMesh( const tnlString& fileName,
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

      if( ! writeHeader( outputFile, mesh ) )
         return false;
   }

   template< typename MeshType >
   static bool writeHeader( ostream& file,
                            MeshType& mesh,
                            bool verbose )
   {
      file << "# vtk DataFile Version 2.0\n";
      file << m_name << "\n";
      file << "ASCII\n";
      file << "DATASET UNSTRUCTURED_GRID\n";
      file << "\n";
   }


};



#endif /* TNLMESHWRITERVTKLEGACY_H_ */
