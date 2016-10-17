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

namespace TNL {

template< typename Index,
          typename Device,
          typename LocalIndex >
class EllpackIndexMultimap;

template< typename Index,
          typename Device,
          typename LocalIndex >
class EllpackIndexMultimapValues
{
   public:
      using DeviceType     = Device;
      using IndexType      = Index;
      using LocalIndexType = LocalIndex;
      using NetworkType    = EllpackIndexMultimap< IndexType, DeviceType, LocalIndexType >;

      EllpackIndexMultimapValues();

      LocalIndexType getPortsCount() const;

      void setOutput( const LocalIndexType& portIndex,
                      const IndexType& output );

      IndexType getOutput( const LocalIndexType& portIndex ) const;

      IndexType& operator[]( const LocalIndexType& portIndex );

      const IndexType& operator[]( const LocalIndexType& portIndex ) const;

      void print( std::ostream& str ) const;

   protected:
      EllpackIndexMultimapValues( IndexType* ports,
                                  const IndexType& input,
                                  const LocalIndexType& portsMaxCount );

      IndexType* ports;

      // TODO: step is unused
//      LocalIndexType step;
      LocalIndexType portsMaxCount;

      friend EllpackIndexMultimap< IndexType, DeviceType, LocalIndexType >;
      friend EllpackIndexMultimap< typename std::remove_const< IndexType >::type, DeviceType, LocalIndexType >;
};

template< typename Index,
          typename Device,
          typename LocalIndex >
std::ostream& operator << ( std::ostream& str, const EllpackIndexMultimapValues< Index, Device, LocalIndex >& ports );

} // namespace TNL

#include <TNL/Experimental/Multimaps/EllpackIndexMultimapValues_impl.h>

