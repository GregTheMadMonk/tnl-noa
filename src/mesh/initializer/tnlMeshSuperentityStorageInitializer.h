/***************************************************************************
                          tnlMeshSuperentityStorageInitializer.h  -  description
                             -------------------
    begin                : Feb 27, 2014
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

#ifndef TNLMESHSUPERENTITYSTORAGEINITIALIZER_H_
#define TNLMESHSUPERENTITYSTORAGEINITIALIZER_H_

#include <mesh/tnlDimensionsTag.h>
#include <algorithm>

template< typename MeshConfig >
class tnlMeshInitializer;

template< typename MeshConfig,
          typename EntityTopology,
          typename DimensionsTag,
          bool SuperentityStorage = tnlMeshSuperentityTraits< MeshConfig, EntityTopology, DimensionsTag::value >::storageEnabled >
class tnlMeshSuperentityStorageInitializerLayer;

template< typename MeshConfig,
          typename EntityTopology >
class tnlMeshSuperentityStorageInitializer :
   public tnlMeshSuperentityStorageInitializerLayer< MeshConfig, EntityTopology, tnlDimensionsTag< tnlMeshTraits< MeshConfig >::meshDimensions > >
{};

template< typename MeshConfig,
          typename EntityTopology,
          typename DimensionsTag >
class tnlMeshSuperentityStorageInitializerLayer< MeshConfig,
                                          EntityTopology,
                                          DimensionsTag,
                                          true >
   : public tnlMeshSuperentityStorageInitializerLayer< MeshConfig,
                                                EntityTopology,
                                                typename DimensionsTag::Decrement >
{
   typedef tnlMeshSuperentityStorageInitializerLayer< MeshConfig,
                                                      EntityTopology,
                                                      typename DimensionsTag::Decrement >      BaseType;

   typedef tnlDimensionsTag< EntityTopology::dimensions >                                    EntityDimensions;
	
   typedef typename tnlMeshTraits< MeshConfig >::GlobalIdArrayType                 GlobalIdArrayType;

      
   typedef typename tnlMeshTraits< MeshConfig >::GlobalIndexType                   GlobalIndexType;
   typedef tnlMeshInitializer< MeshConfig >                                              MeshInitializer;

   public:      
      using BaseType::addSuperentity;
	   
      void addSuperentity( DimensionsTag, GlobalIndexType entityIndex, GlobalIndexType superentityIndex)
      {
           indexPairs.push_back( IndexPair{ entityIndex, superentityIndex } );
      }

      using BaseType::initSuperentities;
      void initSuperentities( MeshInitializer& meshInitializer )
      {
         std::sort( indexPairs.begin(),
                    indexPairs.end(),
                    []( IndexPair pair0, IndexPair pair1 ){ return ( pair0.entityIndex < pair1.entityIndex ); } );

         GlobalIdArrayType &superentityIdsArray = meshInitializer.template meshSuperentityIdsArray< EntityDimensions, DimensionsTag >();
         superentityIdsArray.setSize( static_cast< GlobalIndexType >( indexPairs.size() )  );
         GlobalIndexType currentBegin = 0;
         GlobalIndexType lastEntityIndex = 0;
         //cout << "There are " << superentityIdsArray.getSize() << " superentities..." << endl;
         for( GlobalIndexType i = 0; i < superentityIdsArray.getSize(); i++)
         {
            superentityIdsArray[ i ] = indexPairs[i].superentityIndex;
            
            //cout << "Adding superentity " << indexPairs[i].superentityIndex << " to entity " << lastEntityIndex << endl;
            if( indexPairs[ i ].entityIndex != lastEntityIndex )
            {
               meshInitializer.template superentityIdsArray< DimensionsTag >( meshInitializer.template meshEntitiesArray< EntityDimensions >()[ lastEntityIndex ] ).bind( superentityIdsArray, currentBegin, i - currentBegin );
               currentBegin = i;
               lastEntityIndex = indexPairs[ i ].entityIndex;
            }
         }

         meshInitializer.template superentityIdsArray< DimensionsTag >( meshInitializer.template meshEntitiesArray< EntityDimensions >()[ lastEntityIndex ] ).bind( superentityIdsArray, currentBegin, superentityIdsArray.getSize() - currentBegin );
         indexPairs.clear();

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
class tnlMeshSuperentityStorageInitializerLayer< MeshConfig,
                                          EntityTopology,
                                          DimensionsTag,
                                          false >
   : public tnlMeshSuperentityStorageInitializerLayer< MeshConfig,
                                                EntityTopology,
                                                typename DimensionsTag::Decrement >
{
   typedef tnlMeshSuperentityStorageInitializerLayer< MeshConfig,
                                                EntityTopology,
                                                typename DimensionsTag::Decrement > BaseType;
   typedef tnlMeshInitializer< MeshConfig >                                      MeshInitializerType;
   
   public:
   void addSuperentity()                           {} // This method is due to 'using BaseType::...;' in the derived classes.
   using BaseType::initSuperentities;
   void initSuperentities( MeshInitializerType& ) {cerr << "***" << endl;} 
};

template< typename MeshConfig,
          typename EntityTopology >
class tnlMeshSuperentityStorageInitializerLayer< MeshConfig,
                                          EntityTopology,
                                          tnlDimensionsTag< EntityTopology::dimensions >,
                                          true >
{
   typedef tnlMeshInitializer< MeshConfig >                                      MeshInitializerType;
   
   public:
   void addSuperentity()                           {} // This method is due to 'using BaseType::...;' in the derived classes.
   void initSuperentities( MeshInitializerType& ) {}
};

template< typename MeshConfig,
          typename EntityTopology >
class tnlMeshSuperentityStorageInitializerLayer< MeshConfig,
                                          EntityTopology,
                                          tnlDimensionsTag< EntityTopology::dimensions >,
                                          false >
{
   typedef tnlMeshInitializer< MeshConfig >                                      MeshInitializerType;

   public:
   void addSuperentity()                           {} // This method is due to 'using BaseType::...;' in the derived classes.
   void initSuperentities( MeshInitializerType& ) {}
};




#endif /* TNLMESHSUPERENTITYSTORAGEINITIALIZER_H_ */
