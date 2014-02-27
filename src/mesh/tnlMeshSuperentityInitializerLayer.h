/***************************************************************************
                          tnlMeshSuperentityInitializerLayer.h  -  description
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

#ifndef TNLMESHSUPERENTITYINITIALIZERLAYER_H_
#define TNLMESHSUPERENTITYINITIALIZERLAYER_H_

#include <mesh/traits/tnlStorageTraits.h>
#include <mesh/traits/tnlDimensionsTraits.h>

template< typename ConfigTag,
          typename EntityTag >
class tnlMeshEntityInitializer;

template< typename ConfigTag,
          typename EntityTag,
          typename DimensionsTag,
          typename SuperentityStorageTag = typename tnlMeshSuperentitiesTraits< ConfigTag, EntityTag, DimensionsTag >::SuperentityStorageTag >
class tnlMeshSuperentityInitializerLayer;

template< typename ConfigTag,
          typename EntityTag,
          typename DimensionsTag >
class tnlMeshSuperentityInitializerLayer< ConfigTag,
                                          EntityTag,
                                          DimensionsTag,
                                          tnlStorageTraits< true > >
   : public tnlMeshSuperentityInitializerLayer< ConfigTag,
                                                EntityTag,
                                                typename DimensionsTag::Previous >
{
   typedef tnlMeshSuperentityInitializerLayer< ConfigTag,
                                               EntityTag,
                                               typename DimensionsTag::Previous >      BaseType;

   typedef typename tnlMeshSuperentitiesTraits< ConfigTag,
                                                EntityTag,
                                                DimensionsTag >::GrowableContainerType GrowableContainerType;
   typedef typename GrowableContainerType::ElementType                                 GlobalIndexType;

   typedef tnlMeshEntityInitializer< ConfigTag, EntityTag >                            EntityInitializerType;

   public:

   using BaseType::addSuperentity;
   void addSuperentity( DimensionsTag, GlobalIndexType entityIndex )
   {
      superentityContainer.insert( entityIndex );
   }

   protected:
   void initSuperentities( EntityInitializerType& entityInitializer )
   {
      entityInitializer.superentityContainer( DimensionsTag() ).create( superentityContainer.getSize() );
      superentityContainer.toArray( entityInitializer.superentityContainer( DimensionsTag()) );
      superentityContainer.free();

      BaseType::initSuperentities( entityInitializer );
   }

   private:
   GrowableContainerType superentityContainer;
};

template< typename ConfigTag,
          typename EntityTag,
          typename DimensionsTag >
class tnlMeshSuperentityInitializerLayer< ConfigTag,
                                          EntityTag,
                                          DimensionsTag,
                                          tnlStorageTraits< false > >
   : public tnlMeshSuperentityInitializerLayer< ConfigTag,
                                                EntityTag,
                                                typename DimensionsTag::Previous >
{
};

template< typename ConfigTag,
          typename EntityTag >
class tnlMeshSuperentityInitializerLayer< ConfigTag,
                                          EntityTag,
                                          tnlDimensionsTraits< 0 >,
                                          tnlStorageTraits< true > >
{
};

template< typename ConfigTag,
          typename EntityTag >
class tnlMeshSuperentityInitializerLayer< ConfigTag,
                                          EntityTag,
                                          tnlDimensionsTraits< EntityTag::dimensions >,
                                          tnlStorageTraits< false > >
{
   typedef tnlMeshEntityInitializer< ConfigTag, EntityTag > EntityInitializerType;

   protected:
   void addSuperentity()                           {} // This method is due to 'using BaseType::...;' in the derived classes.
   void initSuperentities( EntityInitializerType& ) {}
};




#endif /* TNLMESHSUPERENTITYINITIALIZERLAYER_H_ */
