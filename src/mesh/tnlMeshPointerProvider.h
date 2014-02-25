/***************************************************************************
                          tnlMeshPointerProvider.h  -  description
                             -------------------
    begin                : Feb 25, 2014
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

#ifndef TNLMESHPOINTERPROVIDER_H_
#define TNLMESHPOINTERPROVIDER_H_

#include<core/tnlAssert.h>

template< typename > class Mesh;

template< typename ConfigTag >
class tnlMeshPointerProvider
{
   typedef Mesh< ConfigTag > MeshType;

   public:

   tnlMeshPointerProvider() : mesh(0) {}

   MeshType& getMesh() const    { tnlAssert( mesh, ); return *mesh; }

   protected:
   void setMesh( MeshType &mesh ) { mesh = &mesh; }

   private:
   MeshType* mesh;
};


#endif /* TNLMESHPOINTERPROVIDER_H_ */
