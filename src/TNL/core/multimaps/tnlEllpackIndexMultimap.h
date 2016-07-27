/***************************************************************************
                          tnlEllpackIndexMultimap.h  -  description
                             -------------------
    begin                : Sep 9, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once 

#include <TNL/Vectors/Vector.h>

namespace TNL {

template< typename Index = int,
          typename Device = Devices::Host >
class tnlEllpackIndexMultimapValues;

template< typename Index = int,
          typename Device = Devices::Host >
class tnlEllpackIndexMultimapConstValues;

template< typename Index = int,
          typename Device = Devices::Host >
class tnlEllpackIndexMultimap
{
   public:
 
      typedef Device                                                       DeviceType;
      typedef Index                                                        IndexType;
      typedef tnlEllpackIndexMultimapValues< IndexType, DeviceType >       ValuesAccessorType;
      typedef tnlEllpackIndexMultimapConstValues< IndexType, DeviceType >  ConstValuesAccessorType;
      typedef Vectors::Vector< IndexType, DeviceType, IndexType >                ValuesAllocationVectorType;
 
      tnlEllpackIndexMultimap();
 
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
 
      Vectors::Vector< IndexType, DeviceType, IndexType > values;
 
      IndexType keysRange, valuesRange, valuesMaxCount;
};

} // namespace TNL

#include <TNL/core/multimaps/tnlEllpackIndexMultimap_impl.h>

