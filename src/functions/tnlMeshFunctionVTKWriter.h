/***************************************************************************
                          tnlMeshFunctionVTKWriter.h  -  description
                             -------------------
    begin                : Jan 28, 2016
    copyright            : (C) 2016 by oberhuber
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

#ifndef TNLMESHFUNCTIONVTKWRITER_H
#define	TNLMESHFUNCTIONVTKWRITER_H

template< typename MeshFunction >
class tnlMeshFunctionVTKWriter
{
   public:
      
      static bool write( const MeshFunction& function,
                         ostream& str )
      {
         std::cerr << "VTK writer for mesh functions defined on mesh type " << MeshFunction::MeshType::getType() << " is not (yet) implmeneted." << std::endl;
         return false;
      }
};


#endif	/* TNLMESHFUNCTIONVTKWRITER_H */

