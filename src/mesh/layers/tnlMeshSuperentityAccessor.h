/***************************************************************************
                          tnlMeshSuperentityAccessor.h  -  description
                             -------------------
    begin                : Sep 11, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLMESHSUPERENTITYACCESSOR_H
#define	TNLMESHSUPERENTITYACCESSOR_H

template< typename IndexMultimapValues >
class tnlMeshSuperentityAccessor
{
   public:
 
      typedef typename IndexMultimapValues::IndexType   GlobalIndexType;
      typedef typename IndexMultimapValues::IndexType   LocalIndexType;
 
      // TODO: Add LocalIndexType to EllpackIndexMultimap
 
      LocalIndexType getSupernetitiesCount() const
      {
         return this->indexes.getPortsCount();
      };
 
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
 
      void print( ostream& str ) const
      {
         str << indexes;
      }
 
   protected:
 
      IndexMultimapValues indexes;
 
};

template< typename IndexMultimapValues >
ostream& operator << ( ostream& str, const tnlMeshSuperentityAccessor< IndexMultimapValues >& superentityAccessor )
{
   superentityAccessor.print( str );
   return str;
}

#endif	/* TNLMESHSUPERENTITYACCESSOR_H */

