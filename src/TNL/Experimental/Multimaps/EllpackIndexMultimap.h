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

template< typename Index = int,
          typename Device = Devices::Host,
          typename LocalIndex = Index >
class EllpackIndexMultimap
{
   public:
      using DeviceType                 = Device;
      using IndexType                  = Index;
      using LocalIndexType             = LocalIndex;
      using ValuesAccessorType         = EllpackIndexMultimapValues< IndexType, DeviceType, LocalIndexType >;
      using ConstValuesAccessorType    = EllpackIndexMultimapValues< const IndexType, DeviceType, LocalIndexType >;
      using ValuesAllocationVectorType = Containers::Vector< LocalIndexType, DeviceType, IndexType >;

      EllpackIndexMultimap();

      static String getType();

      String getTypeVirtual() const;

      void setKeysRange( const IndexType& keysRange );

      const IndexType getKeysRange() const;

      bool allocate( const LocalIndexType& maxValuesCount );

      bool allocate( const ValuesAllocationVectorType& valuesCounts );

      ValuesAccessorType getValues( const IndexType& inputIndex );

      ConstValuesAccessorType getValues( const IndexType& inputIndex ) const;

   protected:
      Containers::Vector< IndexType, DeviceType, IndexType > values;

      IndexType keysRange;
      LocalIndexType maxValuesCount;
};

} // namespace TNL

#include <TNL/Experimental/Multimaps/EllpackIndexMultimap_impl.h>

