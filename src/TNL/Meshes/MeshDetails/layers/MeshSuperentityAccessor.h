/***************************************************************************
                          MeshSuperentityAccessor.h  -  description
                             -------------------
    begin                : Sep 11, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

/***
 * Authors:
 * Oberhuber Tomas, tomas.oberhuber@fjfi.cvut.cz
 * Zabka Vitezslav, zabkav@gmail.com
 */

#pragma once

#include <fstream>

namespace TNL {
namespace Meshes {

template< typename IndexMultimapValues >
class MeshSuperentityAccessor
{
   public:
      typedef typename IndexMultimapValues::IndexType       GlobalIndexType;
      typedef typename IndexMultimapValues::LocalIndexType  LocalIndexType;
 
      LocalIndexType getSupernetitiesCount() const
      {
         return this->indexes.getPortsCount();
      }
 
      void setSuperentityIndex( const LocalIndexType localIndex,
                                const GlobalIndexType globalIndex )
      {
         indexes.setOutput( localIndex, globalIndex );
      }
 
      GlobalIndexType getSuperentityIndex( const LocalIndexType localIndex ) const
      {
         return indexes.getOutput( localIndex );
      }
 
      GlobalIndexType& operator[]( const LocalIndexType localIndex )
      {
         return this->indexes[ localIndex ];
      }
 
      const GlobalIndexType& operator[]( const LocalIndexType localIndex ) const
      {
         return this->indexes[ localIndex ];
      }
 
      void print( std::ostream& str ) const
      {
         str << indexes;
      }
 
   protected:
      IndexMultimapValues indexes;
};

template< typename IndexMultimapValues >
std::ostream& operator << ( std::ostream& str, const MeshSuperentityAccessor< IndexMultimapValues >& superentityAccessor )
{
   superentityAccessor.print( str );
   return str;
}

} // namespace Meshes
} // namespace TNL

