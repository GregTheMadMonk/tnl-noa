/***************************************************************************
                          tnlMeshIntegrityCheckerLayer.h  -  description
                             -------------------
    begin                : Mar 21, 2014
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

#ifndef TNLMESHINTEGRITYCHECKERLAYER_H_
#define TNLMESHINTEGRITYCHECKERLAYER_H_

#include <mesh/traits/tnlMeshEntitiesTraits.h>
#include <mesh/tnlDimensionsTag.h>
#include <mesh/traits/tnlStorageTraits.h>

template< typename MeshType,
          typename DimensionsTag,
          typename EntityStorageTag = typename tnlMeshEntitiesTraits< typename MeshType::Config,
                                                                      DimensionsTag::value >::EntityStorageTag >
class tnlMeshIntegrityCheckerLayer;

template< typename MeshType,
          typename DimensionsTag >
class tnlMeshIntegrityCheckerLayer< MeshType,
                                    DimensionsTag,
                                    tnlStorageTraits< true > >
   : public tnlMeshIntegrityCheckerLayer< MeshType,
                                          typename DimensionsTag::Decrement >
{
   public:
      typedef tnlMeshIntegrityCheckerLayer< MeshType, 
                                            typename DimensionsTag::Decrement >     BaseType;
      enum { dimensions = DimensionsTag::value };

      static bool checkEntities( const MeshType& mesh )
      {         
         typedef typename MeshType::template EntitiesTraits< dimensions >::ContainerType ContainerType;
         typedef typename ContainerType::IndexType                                       GlobalIndexType;
         cout << "Checking entities with " << dimensions << " dimensions ..." << endl;
         for( GlobalIndexType entityIdx = 0;
              entityIdx < mesh.template getNumberOfEntities< dimensions >();
              entityIdx++ )
         {
            cout << "Entity no. " << entityIdx << "               \r" << flush;
         }
         cout << endl;
         if( ! BaseType::checkEntities( mesh ) )
            return false;
         return true;
      }
};

template< typename MeshType >
class tnlMeshIntegrityCheckerLayer< MeshType,
                                    tnlDimensionsTag< 0 >,
                                    tnlStorageTraits< true > >
{
   public:
      enum { dimensions = 0 };

      static bool checkEntities( const MeshType& mesh )
      {
         typedef typename MeshType::template EntitiesTraits< dimensions >::ContainerType ContainerType;
         typedef typename ContainerType::IndexType                                       GlobalIndexType;
         cout << "Checking entities with " << dimensions << " dimensions ..." << endl;
         for( GlobalIndexType entityIdx = 0;
              entityIdx < mesh.template getNumberOfEntities< dimensions >();
              entityIdx++ )
         {
            cout << "Entity no. " << entityIdx << "          \r" << flush;
         }
         cout << endl;
         return true;
      }

};

template< typename MeshType,
          typename DimensionsTag >
class tnlMeshIntegrityCheckerLayer< MeshType,
                                    DimensionsTag,
                                    tnlStorageTraits< false > >
   : public tnlMeshIntegrityCheckerLayer< MeshType,
                                          typename DimensionsTag::Decrement >
{

};

template< typename MeshType >
class tnlMeshIntegrityCheckerLayer< MeshType,
                                    tnlDimensionsTag< 0 >,
                                    tnlStorageTraits< false > >
{

};


#endif /* TNLMESHINTEGRITYCHECKERLAYER_H_ */
