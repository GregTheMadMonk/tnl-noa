/***************************************************************************
                          MeshSuperentityStorageInitializer.h  -  description
                             -------------------
    begin                : Feb 27, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Meshes/MeshDimensionsTag.h>
#include <algorithm>
#include <vector>

namespace TNL {
namespace Meshes {

template< typename MeshConfig >
class MeshInitializer;

template< typename MeshConfig,
          typename EntityTopology,
          typename DimensionsTag,
          bool SuperentityStorage = MeshSuperentityTraits< MeshConfig, EntityTopology, DimensionsTag::value >::storageEnabled >
class MeshSuperentityStorageInitializerLayer;

template< typename MeshConfig,
          typename EntityTopology >
class MeshSuperentityStorageInitializer :
   public MeshSuperentityStorageInitializerLayer< MeshConfig, EntityTopology, MeshDimensionsTag< MeshTraits< MeshConfig >::meshDimensions > >
{};

template< typename MeshConfig,
          typename EntityTopology,
          typename DimensionsTag >
class MeshSuperentityStorageInitializerLayer< MeshConfig,
                                          EntityTopology,
                                          DimensionsTag,
                                          true >
   : public MeshSuperentityStorageInitializerLayer< MeshConfig,
                                                EntityTopology,
                                                typename DimensionsTag::Decrement >
{
   typedef MeshSuperentityStorageInitializerLayer< MeshConfig,
                                                      EntityTopology,
                                                      typename DimensionsTag::Decrement >      BaseType;

   static const int Dimensions = DimensionsTag::value;
   typedef MeshDimensionsTag< EntityTopology::dimensions >                                       EntityDimensions;
	
   typedef MeshTraits< MeshConfig >                                                             MeshTraitsType;
   typedef typename MeshTraitsType::GlobalIdArrayType                                           GlobalIdArrayType; 
   typedef typename MeshTraitsType::GlobalIndexType                                             GlobalIndexType;
   typedef typename MeshTraitsType::LocalIndexType                                              LocalIndexType;
   typedef MeshInitializer< MeshConfig >                                                        MeshInitializerType;
   typedef typename MeshTraitsType::template SuperentityTraits< EntityTopology, Dimensions >    SuperentityTraitsType;
   typedef typename SuperentityTraitsType::StorageNetworkType                                   SuperentityStorageNetwork;

   public:
      using BaseType::addSuperentity;
	
      void addSuperentity( DimensionsTag, GlobalIndexType entityIndex, GlobalIndexType superentityIndex)
      {
         //cout << "Adding superentity with " << DimensionsTag::value << " dimensions of enity with " << EntityDimensions::value << " ... " << std::endl;
         indexPairs.push_back( IndexPair{ entityIndex, superentityIndex } );
      }

      using BaseType::initSuperentities;
      void initSuperentities( MeshInitializerType& meshInitializer )
      {
         throw( 0 ); // TODO: fix this - or it may work with newer version of gcc
         /*std::sort( indexPairs.begin(),
                    indexPairs.end(),
                    []( IndexPair pair0, IndexPair pair1 ){ return ( pair0.entityIndex < pair1.entityIndex ); } );*/

         GlobalIdArrayType &superentityIdsArray = meshInitializer.template meshSuperentityIdsArray< EntityDimensions, DimensionsTag >();
         superentityIdsArray.setSize( static_cast< GlobalIndexType >( indexPairs.size() )  );
         GlobalIndexType currentBegin = 0;
         GlobalIndexType lastEntityIndex = 0;
        std::cout << "There are " << superentityIdsArray.getSize() << " superentities with " << DimensionsTag::value << " dimensions of enities with " << EntityDimensions::value << " ... " << std::endl;
         for( GlobalIndexType i = 0; i < superentityIdsArray.getSize(); i++)
         {
            superentityIdsArray[ i ] = indexPairs[i].superentityIndex;
 
            //cout << "Adding superentity " << indexPairs[i].superentityIndex << " to entity " << lastEntityIndex << std::endl;
            if( indexPairs[ i ].entityIndex != lastEntityIndex )
            {
               meshInitializer.template superentityIdsArray< DimensionsTag >( meshInitializer.template meshEntitiesArray< EntityDimensions >()[ lastEntityIndex ] ).bind( superentityIdsArray, currentBegin, i - currentBegin );
               currentBegin = i;
               lastEntityIndex = indexPairs[ i ].entityIndex;
            }
         }

         meshInitializer.template superentityIdsArray< DimensionsTag >( meshInitializer.template meshEntitiesArray< EntityDimensions >()[ lastEntityIndex ] ).bind( superentityIdsArray, currentBegin, superentityIdsArray.getSize() - currentBegin );
         indexPairs.clear();
 
         /****
          * Network initializer
          */
         SuperentityStorageNetwork& superentityStorageNetwork = meshInitializer.template meshSuperentityStorageNetwork< EntityTopology, DimensionsTag >();
         //GlobalIndexType lastEntityIndex( 0 );
         superentityStorageNetwork.setRanges(
            meshInitializer.template meshEntitiesArray< EntityDimensions >().getSize(),
            meshInitializer.template meshEntitiesArray< DimensionsTag >().getSize() );
         lastEntityIndex = 0;
         typename SuperentityStorageNetwork::ValuesAllocationVectorType storageNetworkAllocationVector;
         storageNetworkAllocationVector.setSize( meshInitializer.template meshEntitiesArray< EntityDimensions >().getSize() );
         storageNetworkAllocationVector.setValue( 0 );
         for( GlobalIndexType i = 0; i < superentityIdsArray.getSize(); i++)
         {
            if( indexPairs[ i ].entityIndex == lastEntityIndex )
               storageNetworkAllocationVector[ lastEntityIndex ]++;
            else
               lastEntityIndex++;
         }
         superentityStorageNetwork.allocate( storageNetworkAllocationVector );
         lastEntityIndex = 0;
         LocalIndexType superentitiesCount( 0 );
         typename SuperentityStorageNetwork::ValuesAccessorType superentitiesIndecis =
            superentityStorageNetwork.getValues( lastEntityIndex );
         for( GlobalIndexType i = 0; i < superentityIdsArray.getSize(); i++)
         {
            if( indexPairs[ i ].entityIndex != lastEntityIndex )
            {
               superentitiesIndecis = superentityStorageNetwork.getValues( ++lastEntityIndex );
               superentitiesCount = 0;
            }
            superentitiesIndecis[ superentitiesCount++ ] =  indexPairs[ i ].superentityIndex;
         }
         BaseType::initSuperentities( meshInitializer );
      }

   private:
      struct IndexPair
      {
         GlobalIndexType entityIndex;
         GlobalIndexType superentityIndex;
      };

      std::vector< IndexPair > indexPairs;
 
};

template< typename MeshConfig,
          typename EntityTopology,
          typename DimensionsTag >
class MeshSuperentityStorageInitializerLayer< MeshConfig,
                                          EntityTopology,
                                          DimensionsTag,
                                          false >
   : public MeshSuperentityStorageInitializerLayer< MeshConfig,
                                                EntityTopology,
                                                typename DimensionsTag::Decrement >
{
   typedef MeshSuperentityStorageInitializerLayer< MeshConfig,
                                                EntityTopology,
                                                typename DimensionsTag::Decrement > BaseType;
   typedef MeshInitializer< MeshConfig >                                      MeshInitializerType;
 
   public:
   void addSuperentity()                           {} // This method is due to 'using BaseType::...;' in the derived classes.
   using BaseType::initSuperentities;
   void initSuperentities( MeshInitializerType& ) { std::cerr << "***" << std::endl;}
};

template< typename MeshConfig,
          typename EntityTopology >
class MeshSuperentityStorageInitializerLayer< MeshConfig,
                                          EntityTopology,
                                          MeshDimensionsTag< EntityTopology::dimensions >,
                                          true >
{
   typedef MeshInitializer< MeshConfig >                                      MeshInitializerType;
 
   public:
   void addSuperentity()                           {} // This method is due to 'using BaseType::...;' in the derived classes.
   void initSuperentities( MeshInitializerType& ) {}
};

template< typename MeshConfig,
          typename EntityTopology >
class MeshSuperentityStorageInitializerLayer< MeshConfig,
                                          EntityTopology,
                                          MeshDimensionsTag< EntityTopology::dimensions >,
                                          false >
{
   typedef MeshInitializer< MeshConfig >                                      MeshInitializerType;

   public:
   void addSuperentity()                           {} // This method is due to 'using BaseType::...;' in the derived classes.
   void initSuperentities( MeshInitializerType& ) {}
};

} // namespace Meshes
} // namespace TNL
