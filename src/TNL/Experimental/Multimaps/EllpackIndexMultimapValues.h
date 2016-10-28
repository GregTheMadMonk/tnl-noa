/***************************************************************************
                          EllpackIndexMultimapValues.h  -  description
                             -------------------
    begin                : Sep 10, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <type_traits>
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

      EllpackIndexMultimapValues( EllpackIndexMultimapValues&& other );

      EllpackIndexMultimapValues& operator=( const EllpackIndexMultimapValues& );

      EllpackIndexMultimapValues& operator=( EllpackIndexMultimapValues&& other );

      void bind( const EllpackIndexMultimapValues& other );

      bool setSize( const LocalIndexType& portsCount );

      LocalIndexType getSize() const;

      LocalIndexType getAllocatedSize() const;

      void setValue( const LocalIndexType& portIndex,
                     const IndexType& value );

      IndexType getValue( const LocalIndexType& portIndex ) const;

      IndexType& operator[]( const LocalIndexType& portIndex );

      const IndexType& operator[]( const LocalIndexType& portIndex ) const;

      bool operator==( const EllpackIndexMultimapValues& other ) const;

      bool operator!=( const EllpackIndexMultimapValues& other ) const;

      void print( std::ostream& str ) const;

   protected:
      using ValuesCountType = typename std::conditional< std::is_const< IndexType >::value,
                                                         typename std::add_const< LocalIndexType >::type,
                                                         LocalIndexType >::type;

      EllpackIndexMultimapValues( IndexType* values,
                                  ValuesCountType* valuesCounts,
                                  const IndexType& input,
                                  const LocalIndexType& allocatedSize );

      IndexType* values;

      ValuesCountType* valuesCount;

      // TODO: this is useless for a const-accessor (without setSize etc.)
      LocalIndexType allocatedSize;

      friend EllpackIndexMultimap< IndexType, DeviceType, LocalIndexType >;
      friend EllpackIndexMultimap< typename std::remove_const< IndexType >::type, DeviceType, LocalIndexType >;
};

template< typename Index,
          typename Device,
          typename LocalIndex >
std::ostream& operator << ( std::ostream& str, const EllpackIndexMultimapValues< Index, Device, LocalIndex >& ports );

} // namespace TNL

#include <TNL/Experimental/Multimaps/EllpackIndexMultimapValues_impl.h>

