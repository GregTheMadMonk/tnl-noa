/***************************************************************************
                          MeshIntegrityCheckerLayer.h  -  description
                             -------------------
    begin                : Mar 21, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Meshes/MeshDetails/traits/MeshEntityTraits.h>
#include <TNL/Meshes/MeshDimensionsTag.h>

namespace TNL {
namespace Meshes {

template< typename MeshType,
          typename DimensionsTag,
          bool EntityStorageTag = MeshEntityTraits< typename MeshType::Config,
                                                       DimensionsTag::value >::storageEnabled >
class MeshIntegrityCheckerLayer;

template< typename MeshType,
          typename DimensionsTag >
class MeshIntegrityCheckerLayer< MeshType,
                                    DimensionsTag,
                                    true >
   : public MeshIntegrityCheckerLayer< MeshType,
                                          typename DimensionsTag::Decrement >
{
   public:
      typedef MeshIntegrityCheckerLayer< MeshType,
                                            typename DimensionsTag::Decrement >     BaseType;
      enum { dimensions = DimensionsTag::value };

      static bool checkEntities( const MeshType& mesh )
      {
         typedef typename MeshType::template EntitiesTraits< dimensions >::ContainerType ContainerType;
         typedef typename ContainerType::IndexType                                       GlobalIndexType;
        std::cout << "Checking entities with " << dimensions << " dimensions ..." << std::endl;
         for( GlobalIndexType entityIdx = 0;
              entityIdx < mesh.template getNumberOfEntities< dimensions >();
              entityIdx++ )
         {
           std::cout << "Entity no. " << entityIdx << "               \r" << std::flush;
         }
        std::cout << std::endl;
         if( ! BaseType::checkEntities( mesh ) )
            return false;
         return true;
      }
};

template< typename MeshType >
class MeshIntegrityCheckerLayer< MeshType,
                                    MeshDimensionsTag< 0 >,
                                    true >
{
   public:
      enum { dimensions = 0 };

      static bool checkEntities( const MeshType& mesh )
      {
         typedef typename MeshType::template EntitiesTraits< dimensions >::ContainerType ContainerType;
         typedef typename ContainerType::IndexType                                       GlobalIndexType;
        std::cout << "Checking entities with " << dimensions << " dimensions ..." << std::endl;
         for( GlobalIndexType entityIdx = 0;
              entityIdx < mesh.template getNumberOfEntities< dimensions >();
              entityIdx++ )
         {
           std::cout << "Entity no. " << entityIdx << "          \r" << std::flush;
         }
        std::cout << std::endl;
         return true;
      }

};

template< typename MeshType,
          typename DimensionsTag >
class MeshIntegrityCheckerLayer< MeshType,
                                    DimensionsTag,
                                    false >
   : public MeshIntegrityCheckerLayer< MeshType,
                                          typename DimensionsTag::Decrement >
{

};

template< typename MeshType >
class MeshIntegrityCheckerLayer< MeshType,
                                    MeshDimensionsTag< 0 >,
                                    false >
{

};

} // namespace Meshes
} // namespace TNL
