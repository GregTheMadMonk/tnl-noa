/***************************************************************************
                          tnlEllpackIndexMultimapValues.h  -  description
                             -------------------
    begin                : Sep 10, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLELLPACKINDEXMULTIMAPVALUES_H
#define	TNLELLPACKINDEXMULTIMAPVALUES_H

#include <ostream>
#include <core/multimaps/tnlEllpackIndexMultimap.h>

template< typename Index,
          typename Device >
class tnlEllpackIndexMultimapValues
{
   public:
 
      typedef Device                                     DeviceType;
      typedef Index                                      IndexType;
      typedef tnlEllpackIndexMultimap< IndexType, DeviceType > NetworkType;
 
      tnlEllpackIndexMultimapValues();
 
      IndexType getPortsCount() const;
 
      void setOutput( const IndexType portIndex,
                      const IndexType output );
 
      IndexType getOutput( const IndexType portIndex ) const;
 
      IndexType& operator[]( const IndexType portIndex );
 
      const IndexType& operator[]( const IndexType portIndex ) const;
 
      void print( std::ostream& str ) const;
 
   protected:
 
      tnlEllpackIndexMultimapValues( IndexType* ports,
                              const IndexType input,
                              const IndexType portsMaxCount );
 
      IndexType* ports;
 
      IndexType step, portsMaxCount;
 
      friend tnlEllpackIndexMultimap< IndexType, DeviceType >;
};

template< typename Index,
          typename Device >
std::ostream& operator << ( std::ostream& str, const tnlEllpackIndexMultimapValues< Index, Device>& ports );

#include <core/multimaps/tnlEllpackIndexMultimapValues_impl.h>


#endif	/* TNLELLPACKINDEXMULTIMAPVALUES_H */

