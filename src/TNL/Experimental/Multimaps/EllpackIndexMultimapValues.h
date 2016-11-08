/***************************************************************************
                          EllpackIndexMultimapValues.h  -  description
                             -------------------
    begin                : Sep 10, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <ostream>
#include <TNL/Experimental/Multimaps/EllpackIndexMultimap.h>

namespace TNL {

template< typename Index,
          typename Device >
class EllpackIndexMultimapValues
{
   public:
 
      typedef Device                                     DeviceType;
      typedef Index                                      IndexType;
      typedef EllpackIndexMultimap< IndexType, DeviceType > NetworkType;
 
      EllpackIndexMultimapValues();
 
      IndexType getPortsCount() const;
 
      void setOutput( const IndexType portIndex,
                      const IndexType output );
 
      IndexType getOutput( const IndexType portIndex ) const;
 
      IndexType& operator[]( const IndexType portIndex );
 
      const IndexType& operator[]( const IndexType portIndex ) const;
 
      void print( std::ostream& str ) const;
 
   protected:
 
      EllpackIndexMultimapValues( IndexType* ports,
                              const IndexType input,
                              const IndexType portsMaxCount );
 
      IndexType* ports;
 
      IndexType step, portsMaxCount;
 
      friend EllpackIndexMultimap< IndexType, DeviceType >;
};

template< typename Index,
          typename Device >
std::ostream& operator << ( std::ostream& str, const EllpackIndexMultimapValues< Index, Device>& ports );

} // namespace TNL

#include <TNL/Experimental/Multimaps/EllpackIndexMultimapValues_impl.h>

