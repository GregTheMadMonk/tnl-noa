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

#include <TNL/Devices/Cuda.h>

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

      __cuda_callable__
      StaticEllpackIndexMultimapValues();

      __cuda_callable__
      StaticEllpackIndexMultimapValues( StaticEllpackIndexMultimapValues&& other );

      __cuda_callable__
      StaticEllpackIndexMultimapValues& operator=( const StaticEllpackIndexMultimapValues& );

      __cuda_callable__
      StaticEllpackIndexMultimapValues& operator=( StaticEllpackIndexMultimapValues&& other );

      __cuda_callable__
      void bind( const StaticEllpackIndexMultimapValues& other );

      constexpr LocalIndexType getSize() const;

      constexpr LocalIndexType getAllocatedSize() const;

      __cuda_callable__
      void setValue( const LocalIndexType& portIndex,
                     const IndexType& value );

      __cuda_callable__
      IndexType getValue( const LocalIndexType& portIndex ) const;

      __cuda_callable__
      IndexType& operator[]( const LocalIndexType& portIndex );

      __cuda_callable__
      const IndexType& operator[]( const LocalIndexType& portIndex ) const;

      __cuda_callable__
      bool operator==( const StaticEllpackIndexMultimapValues& other ) const;

      __cuda_callable__
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

