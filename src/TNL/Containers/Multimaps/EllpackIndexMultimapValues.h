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

#include <TNL/Devices/Cuda.h>

namespace TNL {
namespace Containers {
namespace Multimaps {

template< typename Index,
          typename Device,
          typename LocalIndex,
          int SliceSize >
class EllpackIndexMultimap;

template< typename Index,
          typename Device,
          typename LocalIndex,
          int step = 1 >
class EllpackIndexMultimapValues
{
   public:
      using DeviceType     = Device;
      using IndexType      = Index;
      using LocalIndexType = LocalIndex;
      using NetworkType    = EllpackIndexMultimap< IndexType, DeviceType, LocalIndexType, step >;

      __cuda_callable__
      EllpackIndexMultimapValues();

      __cuda_callable__
      EllpackIndexMultimapValues( EllpackIndexMultimapValues&& other );

      __cuda_callable__
      EllpackIndexMultimapValues& operator=( const EllpackIndexMultimapValues& );

      // converting assignment, needed for 'const int' -> 'int' etc.
      template< typename Index_, typename LocalIndex_, int step_ >
      __cuda_callable__
      EllpackIndexMultimapValues& operator=( const EllpackIndexMultimapValues< Index_, Device, LocalIndex_, step_ >& other );

      __cuda_callable__
      EllpackIndexMultimapValues& operator=( EllpackIndexMultimapValues&& other );

      __cuda_callable__
      void bind( const EllpackIndexMultimapValues& other );

      __cuda_callable__
      void setSize( const LocalIndexType& portsCount );

      __cuda_callable__
      LocalIndexType getSize() const;

      __cuda_callable__
      LocalIndexType getAllocatedSize() const;

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
      bool operator==( const EllpackIndexMultimapValues& other ) const;

      __cuda_callable__
      bool operator!=( const EllpackIndexMultimapValues& other ) const;

      void print( std::ostream& str ) const;

   protected:
      using ValuesCountType = typename std::conditional< std::is_const< IndexType >::value,
                                                         std::add_const_t< LocalIndexType >,
                                                         LocalIndexType >::type;

      __cuda_callable__
      EllpackIndexMultimapValues( IndexType* values,
                                  ValuesCountType* valuesCount,
                                  const LocalIndexType& allocatedSize );

      IndexType* values;

      ValuesCountType* valuesCount;

      // TODO: this is useless for a const-accessor (without setSize etc.)
      LocalIndexType allocatedSize;

      friend EllpackIndexMultimap< IndexType, DeviceType, LocalIndexType, step >;
      friend EllpackIndexMultimap< typename std::remove_const< IndexType >::type, DeviceType, LocalIndexType, step >;
};

template< typename Index,
          typename Device,
          typename LocalIndex,
          int step >
std::ostream& operator << ( std::ostream& str, const EllpackIndexMultimapValues< Index, Device, LocalIndex, step >& ports );

} // namespace Multimaps
} // namespace Containers
} // namespace TNL

#include <TNL/Containers/Multimaps/EllpackIndexMultimapValues.hpp>
