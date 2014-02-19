/***************************************************************************
                          tnl-mesh-convert.h  -  description
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

#ifndef TNL_MESH_CONVERT_H_
#define TNL_MESH_CONVERT_H_

#include <config/tnlParameterContainer.h>
#include <mesh/tnlMeshReaderNetgen.h>
#include <core/mfilename.h>

bool readMesh( const tnlParameterContainer& parameters )
{
   const tnlString& inputFileName = parameters.GetParameter< tnlString >( "input-file" );
   const tnlString fileExt = getFileExtension( inputFileName );
   if( fileExt == "ng" )
   {
      int dimensions;
      if( ! tnlMeshReaderNetgen::detectDimensions( inputFileName ) )
         return false;
      cout << "dimensions = " << dimensions << endl;
      if( ! tnlMeshReaderNetgen::readMesh( inputFileName ) )
         return false;
   }
   return true;
}

#endif /* TNL_MESH_CONVERT_H_ */
