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
#include <TNL/Experimental/Multimaps/StaticEllpackIndexMultimapValues.h>

namespace TNL {

template< int ValuesCount,
          typename Index = int,
          typename Device = Devices::Host,
          typename LocalIndex = Index >
class StaticEllpackIndexMultimap
   : public virtual Object
{
   public:
      using DeviceType                 = Device;
      using IndexType                  = Index;
      using LocalIndexType             = LocalIndex;
      using ValuesAccessorType         = StaticEllpackIndexMultimapValues< ValuesCount, IndexType, DeviceType, LocalIndexType >;
      using ConstValuesAccessorType    = StaticEllpackIndexMultimapValues< ValuesCount, const IndexType, DeviceType, LocalIndexType >;

      StaticEllpackIndexMultimap();

      static String getType();

      String getTypeVirtual() const;

      void setKeysRange( const IndexType& keysRange );

      const IndexType getKeysRange() const;

      bool allocate();

      ValuesAccessorType getValues( const IndexType& inputIndex );

      ConstValuesAccessorType getValues( const IndexType& inputIndex ) const;

      bool operator==( const StaticEllpackIndexMultimap& other ) const;

      bool save( File& file ) const;

      bool load( File& file );

      using Object::load;

      using Object::save;

      void print( std::ostream& str ) const;

   protected:
      Containers::Vector< IndexType, DeviceType, IndexType > values;

      IndexType keysRange;
};

template< int ValuesCount,
          typename Index,
          typename Device,
          typename LocalIndex >
std::ostream& operator << ( std::ostream& str, const StaticEllpackIndexMultimap< ValuesCount, Index, Device, LocalIndex >& multimap );

} // namespace TNL

#include <TNL/Experimental/Multimaps/StaticEllpackIndexMultimap_impl.h>

