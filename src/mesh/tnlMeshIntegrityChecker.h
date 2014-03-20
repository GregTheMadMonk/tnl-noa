/***************************************************************************
                          tnlMeshIntegrityChecker.h  -  description
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

#ifndef TNLMESHINTEGRITYCHECKER_H_
#define TNLMESHINTEGRITYCHECKER_H_

#include<mesh/tnlMesh.h>

template< typename MeshType >
class tnlMeshIntegrityChecker
{
   public:

   typedef typename MeshType::Config                       ConfigTag;
   typedef typename ConfigTag::CellTag                     CellTag;
   typedef tnlDimensionsTraits< CellTag::dimensions >      CellDimensionsTraits;
   typedef tnlMeshEntitiesTraits< ConfigTag,
                                  CellDimensionsTraits >   CellTraits;
   typedef typename CellTraits::SharedContainerType        CellsSharedContainerType;
   typedef tnlDimensionsTraits< 0 >                        VertexDimensionsTraits;
   typedef tnlMeshEntitiesTraits< ConfigTag,
                                  VertexDimensionsTraits > VertexTraits;
   typedef typename VertexTraits::SharedContainerType      VertexSharedConatinerType;

   static bool checkMesh( const MeshType& mesh )
   {
      for( CellsGlobalIndexType cell = 0;
           cell < mesh.getNumberOfCells();
           cell++ )
      {
         cout << "Checking cell number " << cell << endl;
      }
      return true;
   }
};


#endif /* TNLMESHINTEGRITYCHECKER_H_ */
