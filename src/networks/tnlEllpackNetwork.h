/***************************************************************************
                          tnlEllpackNetwork.h  -  description
                             -------------------
    begin                : Sep 9, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TNLELLPACKNETWORK_H
#define	TNLELLPACKNETWORK_H

#include <core/vectors/tnlVector.h>

template< typename Index = int,
          typename Device = tnlHost >
class tnlEllpackNetworkPorts;

template< typename Index = int,
          typename Device = tnlHost >
class tnlEllpackNetworkConstPorts;

template< typename Index = int,
          typename Device = tnlHost >
class tnlEllpackNetwork
{
   public:
      
      typedef Device                                                DeviceType;
      typedef Index                                                 IndexType;
      typedef tnlEllpackNetworkPorts< IndexType, DeviceType >       PortsType;
      typedef tnlEllpackNetworkConstPorts< IndexType, DeviceType >  ConstPortsType;
      typedef tnlVector< IndexType, DeviceType, IndexType >         PortsAllocationVectorType;
            
      tnlEllpackNetwork();
      
      static tnlString getType();

      tnlString getTypeVirtual() const;
      
      void setDimensions( const IndexType inputs,
                          const IndexType outputs );
      
      const IndexType getInputsCount() const;
      
      const IndexType getOutputsCount() const;
      
      void allocatePorts( const PortsAllocationVectorType& portsCount );
      
      PortsType getPorts( const IndexType& inputIndex );
      
      ConstPortsType getPorts( const IndexType& inputIndex ) const;
      
   protected:
      
      tnlVector< IndexType, DeviceType, IndexType > ports;
      
      IndexType inputs, outputs, portsMaxCount;
};

#include <networks/tnlEllpackNetwork_impl.h>

#endif	/* TNLELLPACKNETWORK_H */

