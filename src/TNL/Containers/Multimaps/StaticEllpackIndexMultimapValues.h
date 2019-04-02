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

#include <TNL/TypeTraits.h>
#include <TNL/Devices/Cuda.h>

namespace TNL {
namespace Containers {
namespace Multimaps {

template< int ValuesCount,
          typename Index,
          typename Device,
          typename LocalIndex,
          int step >
class StaticEllpackIndexMultimap;

template< int ValuesCount,
          typename Index,
          typename Device,
          typename LocalIndex,
          int step >
class StaticEllpackIndexMultimapValues
{
   public:
      using DeviceType     = Device;
      using IndexType      = Index;
      using LocalIndexType = LocalIndex;
      using NetworkType    = StaticEllpackIndexMultimap< ValuesCount, IndexType, DeviceType, LocalIndexType, step >;

      __cuda_callable__
      StaticEllpackIndexMultimapValues();

      __cuda_callable__
      StaticEllpackIndexMultimapValues( StaticEllpackIndexMultimapValues&& other );

      __cuda_callable__
      StaticEllpackIndexMultimapValues& operator=( const StaticEllpackIndexMultimapValues& other );

      // converting assignment, needed for 'const int' -> 'int' etc.
      template< typename Index_, typename LocalIndex_, int step_ >
      __cuda_callable__
      StaticEllpackIndexMultimapValues& operator=( const StaticEllpackIndexMultimapValues< ValuesCount, Index_, Device, LocalIndex_, step_ >& other );

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
      __cuda_callable__
      StaticEllpackIndexMultimapValues( IndexType* values );

      IndexType* values;

      friend StaticEllpackIndexMultimap< ValuesCount, IndexType, DeviceType, LocalIndexType, step >;
      friend StaticEllpackIndexMultimap< ValuesCount, typename std::remove_const< IndexType >::type, DeviceType, LocalIndexType, step >;
};

template< int ValuesCount,
          typename Index,
          typename Device,
          typename LocalIndex,
          int step >
std::ostream& operator << ( std::ostream& str, const StaticEllpackIndexMultimapValues< ValuesCount, Index, Device, LocalIndex, step >& ports );

} // namespace Multimaps
} // namespace Containers
} // namespace TNL

#include <TNL/Containers/Multimaps/StaticEllpackIndexMultimapValues_impl.h>
