/***************************************************************************
                          tnlMeshFunctionEvaluator.h  -  description
                             -------------------
    begin                : Jan 1, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
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

#ifndef TNLMESHFUNCTIONEVALUATOR_H
#define	TNLMESHFUNCTIONEVALUATOR_H

#include <mesh/tnlGrid.h>
#include <functions/tnlMeshFunction.h>

template< typename MeshFunction1,
          typename MeshFunction2 >
class tnlMeshFunctionEvaluator
{   
};

template< int Dimensions,
          typename MeshReal1,
          typename MeshReal2,
          typename MeshDevice1,
          typename MeshDevice2,
          typename MeshIndex1,
          typename MeshIndex2,
          int MeshEntityDimensions,
          typename Real >
class tnlMeshFunctionEvaluator< tnlMeshFunction< tnlGrid< Dimensions, MeshReal1, MeshDevice1, MeshIndex1 >, MeshEntityDimensions, Real >,
                                tnlMeshFunction< tnlGrid< Dimensions, MeshReal2, MeshDevice2, MeshIndex2 >, MeshEntityDimensions, Real > >
{
   public:
      
      typedef tnlGrid< Dimensions, MeshReal1, MeshDevice1, MeshIndex1 > Mesh1;
      typedef tnlGrid< Dimensions, MeshReal2, MeshDevice2, MeshIndex2 > Mesh2;
      typedef tnlMeshFunction< Mesh1, MeshEntityDimensions, Real > MeshFunction1;
      typedef tnlMeshFunction< Mesh2, MeshEntityDimensions, Real > MeshFunction2;
      
      static void assign( const MeshFunction1& f1,
                          MeshFunction2& f2 )
      {
         if( f1.getMesh().getDimensions() == f2.getMesh().getDimensions() )
            f2.getData() = f1.getData();
         else
         {
            //TODO: Interpolace
         }
      };
};

#endif	/* TNLMESHFUNCTIONEVALUATOR_H */

