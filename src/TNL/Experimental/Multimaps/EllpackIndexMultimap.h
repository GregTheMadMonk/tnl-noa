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

namespace TNL {

template< typename Index = int,
          typename Device = Devices::Host >
class EllpackIndexMultimapValues;

template< typename Index = int,
          typename Device = Devices::Host >
class EllpackIndexMultimapConstValues;

template< typename Index = int,
          typename Device = Devices::Host >
class EllpackIndexMultimap
{
   public:
 
      typedef Device                                                       DeviceType;
      typedef Index                                                        IndexType;
      typedef EllpackIndexMultimapValues< IndexType, DeviceType >       ValuesAccessorType;
      typedef EllpackIndexMultimapConstValues< IndexType, DeviceType >  ConstValuesAccessorType;
      typedef Containers::Vector< IndexType, DeviceType, IndexType >                ValuesAllocationVectorType;
 
      EllpackIndexMultimap();
 
      static String getType();

      String getTypeVirtual() const;
 
      void setRanges( const IndexType keysRange,
                      const IndexType valuesRange );
 
      const IndexType getKeysRange() const;
 
      const IndexType getValuesRange() const;
 
      void allocate( const ValuesAllocationVectorType& portsCount );
 
      ValuesAccessorType getValues( const IndexType& inputIndex );
 
      ConstValuesAccessorType getValues( const IndexType& inputIndex ) const;
 
   protected:
 
      Containers::Vector< IndexType, DeviceType, IndexType > values;
 
      IndexType keysRange, valuesRange, valuesMaxCount;
};

} // namespace TNL

#include <TNL/Experimental/Multimaps/EllpackIndexMultimap_impl.h>

