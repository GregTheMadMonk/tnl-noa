/***************************************************************************
                          tnlMeshSuperentityAccess.h  -  description
                             -------------------
    begin                : Aug 15, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
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

#ifndef TNLSUPERENTITYACCESS_H
#define	TNLSUPERENTITYACCESS_H

#include <mesh/traits/tnlStorageTraits.h>
#include <mesh/traits/tnlMeshConfigTraits.h>


template< typename MeshConfig,
          typename MeshEntity,
          typename Dimensions,
          typename SuperentityStorage = 
             tnlStorageTraits< tnlMeshConfigTraits< MeshConfig >::template SuperentityTraits< MeshEntity, Dimensions>::storageEnabled > >
class tnlMeshSuperentityAccessLayer;


template< typename MeshConfig,
          typename MeshEntity >
class tnlMeshSuperentityAccess :
   public tnlMeshSuperentityAccessLayer< MeshConfig, 
                                         MeshEntity,
                                         tnlDimensionsTag< tnlMeshConfigTraits< MeshConfig >::meshDimensions > >
{
   public:
      bool operator == ( const tnlMeshSuperentityAccess< MeshConfig, MeshEntity>& a ) const { return true; } // TODO: fix
      
      void print( ostream& str ) const{};

};

template< typename MeshConfig,
          typename MeshEntity,
          typename Dimensions >
class tnlMeshSuperentityAccessLayer< MeshConfig,
                                     MeshEntity,
                                     Dimensions,
                                     tnlStorageTraits< true > > :
   public tnlMeshSuperentityAccessLayer< MeshConfig, MeshEntity, typename Dimensions::Decrement >
{
	typedef tnlMeshSuperentityAccessLayer< MeshConfig, MeshEntity, typename Dimensions::Decrement > BaseType;

   public:
	   typedef typename tnlMeshConfigTraits< MeshConfig >::IdArrayAccessorType    IdArrayAccessorType;

	   using BaseType::superentityIds;
	   IdArrayAccessorType superentityIds( Dimensions ) const { return m_superentityIndices; }

	   using BaseType::superentityIdsArray;
	   IdArrayAccessorType &superentityIdsArray( Dimensions ) { return m_superentityIndices; }
      
      //bool operator == ( const tnlMeshSuperentityAccessLayer< MeshConfig, MeshEntity, Dimensions, tnlStorageTraits< true > >& l ) { return true; } // TODO: fix

   private:
	   IdArrayAccessorType m_superentityIndices;
};

template< typename MeshConfig,
          typename MeshEntity,
          typename Dimensions >
class tnlMeshSuperentityAccessLayer< MeshConfig,
                                     MeshEntity,
                                     Dimensions,
                                     tnlStorageTraits< false > > :
   public tnlMeshSuperentityAccessLayer< MeshConfig, MeshEntity, typename Dimensions::Decrement >
{
};

template< typename MeshConfig,
          typename MeshEntity >
class tnlMeshSuperentityAccessLayer< MeshConfig,
                                     MeshEntity,
                                     tnlDimensionsTag< MeshEntity::dimensions >,
                                     tnlStorageTraits< false > >
{
   protected:
	   /***
       * Necessary because of 'using TBase::...;' in the derived classes
       */
	   void superentityIds()      {}
	   void superentityIdsArray() {}
};

template< typename MeshConfig,
          typename MeshEntity >
class tnlMeshSuperentityAccessLayer< MeshConfig,
                                     MeshEntity,
                                     tnlDimensionsTag< MeshEntity::dimensions >,
                                     tnlStorageTraits< true > >
{
   protected:
	   /***
       * Necessary because of 'using TBase::...;' in the derived classes
       */
	   void superentityIds()      {}
	   void superentityIdsArray() {}
};


#endif	/* TNLSUPERENTITYACCESS_H */

