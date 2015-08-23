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

#include <mesh/traits/tnlStorageTraits.h>
#include <mesh/tnlDimensionsTag.h>
#include <algorithm>

template< typename ConfigTag >
class tnlMeshInitializer;

template< typename ConfigTag,
          typename EntityTag,
          typename DimensionsTag,
          typename SuperentityStorageTag = typename tnlMeshSuperentitiesTraits< ConfigTag, EntityTag, DimensionsTag >::SuperentityStorageTag >
class tnlMeshSuperentityStorageInitializerLayer;

template< typename ConfigTag,
          typename EntityTag >
class tnlMeshSuperentityStorageInitializer :
   public tnlMeshSuperentityStorageInitializerLayer< ConfigTag, EntityTag, tnlDimensionsTag< tnlMeshConfigTraits< ConfigTag >::meshDimensions > >
{};

template< typename ConfigTag,
          typename EntityTag,
          typename DimensionsTag >
class tnlMeshSuperentityStorageInitializerLayer< ConfigTag,
                                          EntityTag,
                                          DimensionsTag,
                                          tnlStorageTraits< true > >
   : public tnlMeshSuperentityStorageInitializerLayer< ConfigTag,
                                                EntityTag,
                                                typename DimensionsTag::Decrement >
{
   typedef tnlMeshSuperentityStorageInitializerLayer< ConfigTag,
                                                      EntityTag,
                                                      typename DimensionsTag::Decrement >      BaseType;

   typedef tnlDimensionsTag< EntityTag::dimensions >                                    EntityDimensions;
	
   typedef typename tnlMeshConfigTraits< ConfigTag >::GlobalIdArrayType                 GlobalIdArrayType;

      
   typedef typename tnlMeshConfigTraits< ConfigTag >::GlobalIndexType                   GlobalIndexType;
   typedef tnlMeshInitializer< ConfigTag >                                              MeshInitializer;

   public:      
      using BaseType::addSuperentity;
	   
      void addSuperentity( DimensionsTag, GlobalIndexType entityIndex, GlobalIndexType superentityIndex)
      {
           indexPairs.push_back( IndexPair{ entityIndex, superentityIndex } );
      }

      using BaseType::initSuperentities;
      void initSuperentities( MeshInitializer& meshInitializer )
      {
         cerr << "####" << endl;
         std::sort( indexPairs.begin(),
                    indexPairs.end(),
                    []( IndexPair pair0, IndexPair pair1 ){ return ( pair0.entityIndex < pair1.entityIndex ); } );

         GlobalIdArrayType &superentityIdsArray = meshInitializer.template meshSuperentityIdsArray< EntityDimensions, DimensionsTag >();
         superentityIdsArray.setSize( static_cast< GlobalIndexType >( indexPairs.size() )  );
         GlobalIndexType currentBegin = 0;
         GlobalIndexType lastEntityIndex = 0;
         cout << "There are " << superentityIdsArray.getSize() << " superentities..." << endl;
         for( GlobalIndexType i = 0; i < superentityIdsArray.getSize(); i++)
         {
            superentityIdsArray[ i ] = indexPairs[i].superentityIndex;
            
            cout << "Adding superentity " << indexPairs[i].superentityIndex << " to entity " << lastEntityIndex << endl;
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

template< typename ConfigTag,
          typename EntityTag,
          typename DimensionsTag >
class tnlMeshSuperentityStorageInitializerLayer< ConfigTag,
                                          EntityTag,
                                          DimensionsTag,
                                          tnlStorageTraits< false > >
   : public tnlMeshSuperentityStorageInitializerLayer< ConfigTag,
                                                EntityTag,
                                                typename DimensionsTag::Decrement >
{
   typedef tnlMeshSuperentityStorageInitializerLayer< ConfigTag,
                                                EntityTag,
                                                typename DimensionsTag::Decrement > BaseType;
   typedef tnlMeshInitializer< ConfigTag >                                      MeshInitializerType;
   
   public:
   void addSuperentity()                           {} // This method is due to 'using BaseType::...;' in the derived classes.
   using BaseType::initSuperentities;
   void initSuperentities( MeshInitializerType& ) {cerr << "***" << endl;} 
};

template< typename ConfigTag,
          typename EntityTag >
class tnlMeshSuperentityStorageInitializerLayer< ConfigTag,
                                          EntityTag,
                                          tnlDimensionsTag< EntityTag::dimensions >,
                                          tnlStorageTraits< true > >
{
   typedef tnlMeshInitializer< ConfigTag >                                      MeshInitializerType;
   
   public:
   void addSuperentity()                           {} // This method is due to 'using BaseType::...;' in the derived classes.
   void initSuperentities( MeshInitializerType& ) {cerr << "***" << endl;}
};

template< typename ConfigTag,
          typename EntityTag >
class tnlMeshSuperentityStorageInitializerLayer< ConfigTag,
                                          EntityTag,
                                          tnlDimensionsTag< EntityTag::dimensions >,
                                          tnlStorageTraits< false > >
{
   typedef tnlMeshInitializer< ConfigTag >                                      MeshInitializerType;

   public:
   void addSuperentity()                           {} // This method is due to 'using BaseType::...;' in the derived classes.
   void initSuperentities( MeshInitializerType& ) { cerr << "***" << endl;}
};




#endif /* TNLMESHSUPERENTITYSTORAGEINITIALIZER_H_ */
