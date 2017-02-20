/***************************************************************************
                          EllpackIndexMultimap.h  -  description
                             -------------------
    begin                : Sep 9, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Containers/Vector.h>
#include <TNL/Experimental/Multimaps/EllpackIndexMultimapValues.h>

namespace TNL {

template< typename Device >
struct EllpackIndexMultimapSliceSizeGetter
{
   static constexpr int SliceSize = 1;
};

template<>
struct EllpackIndexMultimapSliceSizeGetter< Devices::Cuda >
{
   static constexpr int SliceSize = 32;
};

template< typename Index = int,
          typename Device = Devices::Host,
          typename LocalIndex = Index,
          int SliceSize = EllpackIndexMultimapSliceSizeGetter< Device >::SliceSize >
class EllpackIndexMultimap
   : public virtual Object
{
   public:
      using DeviceType                 = Device;
      using IndexType                  = Index;
      using LocalIndexType             = LocalIndex;
      using ValuesAccessorType         = EllpackIndexMultimapValues< IndexType, DeviceType, LocalIndexType, SliceSize >;
      using ConstValuesAccessorType    = EllpackIndexMultimapValues< const IndexType, DeviceType, LocalIndexType, SliceSize >;
      using ValuesAllocationVectorType = Containers::Vector< LocalIndexType, DeviceType, IndexType >;

      EllpackIndexMultimap() = default;

      template< typename Device_ >
      EllpackIndexMultimap( const EllpackIndexMultimap< Index, Device_, LocalIndex, SliceSize >& other );

      template< typename Device_ >
      EllpackIndexMultimap& operator=( const EllpackIndexMultimap< Index, Device_, LocalIndex, SliceSize >& other );

      static String getType();

      String getTypeVirtual() const;

      void setKeysRange( const IndexType& keysRange );

      __cuda_callable__
      const IndexType getKeysRange() const;

      bool allocate( const LocalIndexType& maxValuesCount );

      bool allocate( const ValuesAllocationVectorType& valuesCounts );

      template< typename Device_, int SliceSize_ >
      bool setLike( const EllpackIndexMultimap< Index, Device_, LocalIndex, SliceSize_ >& other );

      __cuda_callable__
      ValuesAccessorType getValues( const IndexType& inputIndex );

      __cuda_callable__
      ConstValuesAccessorType getValues( const IndexType& inputIndex ) const;

      bool operator==( const EllpackIndexMultimap& other ) const;

      bool save( File& file ) const;

      bool load( File& file );

      using Object::load;

      using Object::save;

      void print( std::ostream& str ) const;

   protected:
      Containers::Vector< IndexType, DeviceType, IndexType > values;
      Containers::Vector< LocalIndexType, DeviceType, IndexType > valuesCounts;

      IndexType keysRange = 0;
      LocalIndexType maxValuesCount = 0;

      __cuda_callable__
      IndexType getAllocationKeysRange( IndexType keysRange ) const;

      // friend class is needed for templated assignment operators
      template< typename Index_, typename Device_, typename LocalIndex_, int SliceSize_ >
      friend class EllpackIndexMultimap;
};

template< typename Index,
          typename Device,
          typename LocalIndex,
          int SliceSize >
std::ostream& operator << ( std::ostream& str, const EllpackIndexMultimap< Index, Device, LocalIndex, SliceSize >& multimap );

} // namespace TNL

#include <TNL/Experimental/Multimaps/EllpackIndexMultimap_impl.h>

