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
#include <mesh/traits/tnlDimensionsTraits.h>
#include <mesh/traits/tnlStorageTraits.h>

template< typename MeshType,
          typename DimensionsTraits,
          typename EntityStorageTag = typename tnlMeshEntitiesTraits< typename MeshType::Config,
                                                                      DimensionsTraits >::EntityStorageTag >
class tnlMeshIntegrityCheckerLayer;

template< typename MeshType,
          typename DimensionsTraits >
class tnlMeshIntegrityCheckerLayer< MeshType,
                                    DimensionsTraits,
                                    tnlStorageTraits< true > >
   : public tnlMeshIntegrityCheckerLayer< MeshType,
                                          typename DimensionsTraits::Previous >
{
   public:
      typedef tnlMeshIntegrityCheckerLayer< MeshType, 
                                            typename DimensionsTraits::Previous >     BaseType;
      enum { dimensions = DimensionsTraits::value };

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
                                    tnlDimensionsTraits< 0 >,
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
          typename DimensionsTraits >
class tnlMeshIntegrityCheckerLayer< MeshType,
                                    DimensionsTraits,
                                    tnlStorageTraits< false > >
   : public tnlMeshIntegrityCheckerLayer< MeshType,
                                          typename DimensionsTraits::Previous >
{

};

template< typename MeshType >
class tnlMeshIntegrityCheckerLayer< MeshType,
                                    tnlDimensionsTraits< 0 >,
                                    tnlStorageTraits< false > >
{

};


#endif /* TNLMESHINTEGRITYCHECKERLAYER_H_ */
