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

#include <networks/tnlEllpackNetworkPorts.h>

template< typename Device = tnlHost,
          typename Index = int >
class tnlEllpackNetworkConstPorts;

template< typename Device = tnlHost,
          typename Index = int >
class tnlEllpackNetwork
{
   public:
      
      typedef Device                                                DeviceType;
      typedef Index                                                 IndexType;
      typedef tnlEllpackNetworkPorts< DeviceType, IndexType >       PortsType;
      typedef tnlEllpackNetworkConstPorts< DeviceType, IndexType >  ConstPortsType;
      typedef tnlVector< IndexType, DeviceType, IndexType >         PortsAllocationVectorType;
            
      tnlEllpackNetwork();
      
      void setDimensions( const IndexType inputs,
                          const IndexType outputs );
      
      const IndexType getInputsCount() const;
      
      const IndexType getOutputsCount() const;
      
      void allocatePorts( const PortsAllocationVectorType& portsCount );
      
      PortsType getPorts( const IndexType& inputIndex );
      
      ConstPortsType getPorts( const IndexType& inputIndex ) const;
      
   protected:
      
      tnlVector< IndexType, DeviceType, IndexType > links;
      
      IndexType inputs, outputs, portsMaxCount;
};

#include <networks/tnlEllpackNetworks_impl.h>

#endif	/* TNLELLPACKNETWORK_H */

