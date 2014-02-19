/***************************************************************************
                          tnlMeshReaderNetgen.h  -  description
                             -------------------
    begin                : Feb 19, 2014
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

#ifndef TNLMESHREADERNETGEN_H_
#define TNLMESHREADERNETGEN_H_

#include <fstream>
#include <istream>
#include <sstream>

class tnlMeshReaderNetgen
{
   public:

   static int detectDimensions( const tnlString& fileName )
   {
      fstream inputFile( fileName.getString() );
      if( ! inputFile )
      {
         cerr << "I am not able to open the file " << fileName << "." << endl;
         return false;
      }
      
      std::string line;
      int count;
      std::istringstream iss;

   }

   static bool readMesh( const tnlString& fileName )
   {
   }

   protected:


};


#endif /* TNLMESHREADERNETGEN_H_ */
