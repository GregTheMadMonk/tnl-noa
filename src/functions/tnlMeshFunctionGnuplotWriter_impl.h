/***************************************************************************
                          tnlMeshFunctionGnuplotWriter.h  -  description
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

#ifndef TNLMESHFUNCTIONGNUPLOTWRITER_IMPL_H
#define	TNLMESHFUNCTIONGNUPLOTWRITER_IMPL_H

template< typename MeshFunction >
tnlMeshFunctionGnuplotWriter< MeshFunction >::
bool
write( const MeshFunction& function,
       ostream& str )
{
   std::cerr << "Gnuplot writer for mesh functions defined on mesh type " << MeshFunction::Mesh::getType() << " is not (yet) implmeneted." << std::endl;
   return false;   
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real >
bool
tnlMeshFunctionGnuplotWriter< tnlMeshFunction< tnlGrid< 1, MeshReal, Device, MeshIndex >, 1, Real >::
write( const MeshFunctionType& function,
       ostream& str )
{
}


#endif	/* TNLMESHFUNCTIONGNUPLOTWRITER_IMPL_H */

