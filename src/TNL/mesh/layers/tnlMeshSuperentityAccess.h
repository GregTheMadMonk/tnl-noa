/***************************************************************************
                          tnlMeshSuperentityAccess.h  -  description
                             -------------------
    begin                : Aug 15, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/mesh/traits/tnlMeshTraits.h>

namespace TNL {

template< typename MeshConfig,
          typename MeshEntity,
          typename DimensionsTag,
          bool SuperentityStorage =
             tnlMeshTraits< MeshConfig >::template SuperentityTraits< MeshEntity, DimensionsTag::value >::storageEnabled >
class tnlMeshSuperentityAccessLayer;


template< typename MeshConfig,
          typename MeshEntity >
class tnlMeshSuperentityAccess :
   public tnlMeshSuperentityAccessLayer< MeshConfig,
                                         MeshEntity,
                                         tnlDimensionsTag< tnlMeshTraits< MeshConfig >::meshDimensions > >
{
   public:
      typedef tnlMeshSuperentityAccessLayer< MeshConfig,
                                             MeshEntity,
                                             tnlDimensionsTag< tnlMeshTraits< MeshConfig >::meshDimensions > > BaseType;
 
      bool operator == ( const tnlMeshSuperentityAccess< MeshConfig, MeshEntity>& a ) const { return true; } // TODO: fix
 
      void print( std::ostream& str ) const
      {
         BaseType::print( str );
      };

};

template< typename MeshConfig,
          typename MeshEntity,
          typename Dimensions >
class tnlMeshSuperentityAccessLayer< MeshConfig,
                                     MeshEntity,
                                     Dimensions,
                                     true > :
   public tnlMeshSuperentityAccessLayer< MeshConfig, MeshEntity, typename Dimensions::Decrement >
{
	typedef tnlMeshSuperentityAccessLayer< MeshConfig, MeshEntity, typename Dimensions::Decrement > BaseType;

   public:
 
      typedef tnlMeshTraits< MeshConfig >                                   MeshTraits;
      typedef typename MeshTraits::template SuperentityTraits< MeshEntity, Dimensions::value > SuperentityTraits;
	   typedef typename MeshTraits::IdArrayAccessorType                          IdArrayAccessorType;
      typedef typename SuperentityTraits::StorageNetworkType                    StorageNetworkType;
      typedef typename SuperentityTraits::SuperentityAccessorType               SuperentityAccessorType;
      //typedef typename StorageNetworkType::PortsType                            SuperentityAccessorType;

	   using BaseType::superentityIds;
	   IdArrayAccessorType superentityIds( Dimensions ) const { return m_superentityIndices; }

	   using BaseType::superentityIdsArray;
	   IdArrayAccessorType &superentityIdsArray( Dimensions ) { return m_superentityIndices; }
 
      using BaseType::getSuperentityIndices;
      const SuperentityAccessorType& getSuperentityIndices( Dimensions ) const
      {
         std::cerr << "###" << std::endl;
         return this->superentityIndices;
      }
 
      SuperentityAccessorType& getSuperentityIndices( Dimensions )
      {
         std::cerr << "######" << std::endl;
         return this->superentityIndices;
      }
 
      void print( std::ostream& str ) const
      {
         str << "Superentities with " << Dimensions::value << " dimensions are: " <<
            this->superentityIndices << std::endl;
         BaseType::print( str );
      }
 
      //bool operator == ( const tnlMeshSuperentityAccessLayer< MeshConfig, MeshEntity, Dimensions, tnlStorageTraits< true > >& l ) { return true; } // TODO: fix

   private:
	   IdArrayAccessorType m_superentityIndices;
 
      SuperentityAccessorType superentityIndices;
 
};

template< typename MeshConfig,
          typename MeshEntity,
          typename Dimensions >
class tnlMeshSuperentityAccessLayer< MeshConfig,
                                     MeshEntity,
                                     Dimensions,
                                     false > :
   public tnlMeshSuperentityAccessLayer< MeshConfig, MeshEntity, typename Dimensions::Decrement >
{
};

template< typename MeshConfig,
          typename MeshEntity >
class tnlMeshSuperentityAccessLayer< MeshConfig,
                                     MeshEntity,
                                     tnlDimensionsTag< MeshEntity::dimensions >,
                                     false >
{
   protected:
	   /***
       * Necessary because of 'using TBase::...;' in the derived classes
       */
	   void superentityIds()      {}
	   void superentityIdsArray() {}
 
      void getSuperentityIndices() {};
 
      void print( std::ostream& str ) const {};
};

template< typename MeshConfig,
          typename MeshEntity >
class tnlMeshSuperentityAccessLayer< MeshConfig,
                                     MeshEntity,
                                     tnlDimensionsTag< MeshEntity::dimensions >,
                                     true >
{
   protected:
	   /***
       * Necessary because of 'using TBase::...;' in the derived classes
       */
	   void superentityIds()      {}
	   void superentityIdsArray() {}
 
      void getSuperentityIndices() {};
 
      void print( std::ostream& str ) const {};
};

} // namespace TNL

