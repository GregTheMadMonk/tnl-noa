/***************************************************************************
                          StaticEllpackIndexMultimapValues.h  -  description
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

template< int ValuesCount,
          typename Index,
          typename Device,
          typename LocalIndex >
class StaticEllpackIndexMultimap;

template< int ValuesCount,
          typename Index,
          typename Device,
          typename LocalIndex >
class StaticEllpackIndexMultimapValues
{
   public:
      using DeviceType     = Device;
      using IndexType      = Index;
      using LocalIndexType = LocalIndex;
      using NetworkType    = StaticEllpackIndexMultimap< ValuesCount, IndexType, DeviceType, LocalIndexType >;

      StaticEllpackIndexMultimapValues();

      StaticEllpackIndexMultimapValues( StaticEllpackIndexMultimapValues&& other );

      StaticEllpackIndexMultimapValues& operator=( const StaticEllpackIndexMultimapValues& );

      StaticEllpackIndexMultimapValues& operator=( StaticEllpackIndexMultimapValues&& other );

      void bind( const StaticEllpackIndexMultimapValues& other );

      constexpr LocalIndexType getSize() const;

      constexpr LocalIndexType getAllocatedSize() const;

      void setValue( const LocalIndexType& portIndex,
                     const IndexType& value );

      IndexType getValue( const LocalIndexType& portIndex ) const;

      IndexType& operator[]( const LocalIndexType& portIndex );

      const IndexType& operator[]( const LocalIndexType& portIndex ) const;

      bool operator==( const StaticEllpackIndexMultimapValues& other ) const;

      bool operator!=( const StaticEllpackIndexMultimapValues& other ) const;

      void print( std::ostream& str ) const;

   protected:
      StaticEllpackIndexMultimapValues( IndexType* values,
                                        const IndexType& input );

      IndexType* values;

      friend StaticEllpackIndexMultimap< ValuesCount, IndexType, DeviceType, LocalIndexType >;
      friend StaticEllpackIndexMultimap< ValuesCount, typename std::remove_const< IndexType >::type, DeviceType, LocalIndexType >;
};

template< int ValuesCount,
          typename Index,
          typename Device,
          typename LocalIndex >
std::ostream& operator << ( std::ostream& str, const StaticEllpackIndexMultimapValues< ValuesCount, Index, Device, LocalIndex >& ports );

} // namespace TNL

#include <TNL/Experimental/Multimaps/StaticEllpackIndexMultimapValues_impl.h>

