/***************************************************************************
                          StaticEllpackIndexMultimap.h  -  description
                             -------------------
    begin                : Sep 9, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Containers/Vector.h>
#include <TNL/Containers/Multimaps/StaticEllpackIndexMultimapValues.h>

namespace TNL {

template< typename Device >
struct StaticEllpackIndexMultimapSliceSizeGetter
{
   static constexpr int SliceSize = 1;
};

template<>
struct StaticEllpackIndexMultimapSliceSizeGetter< Devices::Cuda >
{
   static constexpr int SliceSize = 32;
};

template< int ValuesCount,
          typename Index = int,
          typename Device = Devices::Host,
          typename LocalIndex = Index,
          int SliceSize = StaticEllpackIndexMultimapSliceSizeGetter< Device >::SliceSize >
class StaticEllpackIndexMultimap
   : public Object
{
   public:
      using DeviceType                 = Device;
      using IndexType                  = Index;
      using LocalIndexType             = LocalIndex;
      using ValuesAccessorType         = StaticEllpackIndexMultimapValues< ValuesCount, IndexType, DeviceType, LocalIndexType, SliceSize >;
      using ConstValuesAccessorType    = StaticEllpackIndexMultimapValues< ValuesCount, const IndexType, DeviceType, LocalIndexType, SliceSize >;

      StaticEllpackIndexMultimap() = default;

      template< typename Device_ >
      StaticEllpackIndexMultimap( const StaticEllpackIndexMultimap< ValuesCount, Index, Device_, LocalIndex, SliceSize >& other );

      template< typename Device_ >
      StaticEllpackIndexMultimap& operator=( const StaticEllpackIndexMultimap< ValuesCount, Index, Device_, LocalIndex, SliceSize >& other );

      static String getType();

      String getTypeVirtual() const;

      void setKeysRange( const IndexType& keysRange );

      __cuda_callable__
      const IndexType getKeysRange() const;

      void allocate();

      template< typename Device_ >
      void setLike( const StaticEllpackIndexMultimap< ValuesCount, Index, Device_, LocalIndex, SliceSize >& other );

      __cuda_callable__
      ValuesAccessorType getValues( const IndexType& inputIndex );

      __cuda_callable__
      ConstValuesAccessorType getValues( const IndexType& inputIndex ) const;

      bool operator==( const StaticEllpackIndexMultimap& other ) const;

      bool save( File& file ) const;

      bool load( File& file );

      using Object::load;

      using Object::save;

      void print( std::ostream& str ) const;

   protected:
      Containers::Vector< IndexType, DeviceType, IndexType > values;

      IndexType keysRange = 0;

      __cuda_callable__
      IndexType getAllocationKeysRange( IndexType keysRange ) const;

      // friend class is needed for templated assignment operators
      template< int ValuesCount_, typename Index_, typename Device_, typename LocalIndex_, int SliceSize_ >
      friend class StaticEllpackIndexMultimap;
};

template< int ValuesCount,
          typename Index,
          typename Device,
          typename LocalIndex,
          int SliceSize >
std::ostream& operator << ( std::ostream& str, const StaticEllpackIndexMultimap< ValuesCount, Index, Device, LocalIndex, SliceSize >& multimap );

} // namespace TNL

#include <TNL/Containers/Multimaps/StaticEllpackIndexMultimap_impl.h>

