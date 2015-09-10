/***************************************************************************
                          tnlEllpackNetworkPorts.h  -  description
                             -------------------
    begin                : Sep 10, 2015
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

#ifndef TNLELLPACKNETWORKPORTS_H
#define	TNLELLPACKNETWORKPORTS_H

template< typename Device,
          typename Index >
class tnlEllpackNetworkPorts
{
   public:
      
      typedef Device                                     DeviceType;
      typedef Index                                      IndexType;
      typedef tnlEllpackNetwork< DeviceType, IndexType > NetworkType;
      
      void setOutput( const IndexType portIndex,
                      const IndexType output );
      
      IndexType getOutput( const IndexType portIndex ) const;
      
      IndexType& operator[]( const IndexType portIndex );
      
      const IndexType& operator[]( const IndexType portIndex ) const;
      
   protected:
      
      tnlEllpackNetworkPorts( IndexType* ports, 
                              const IndexType input,
                              const IndexType portsMaxCount );
      
      IndexType* ports;
      
      IndexType step;
      
      friend tnlEllpackNetwork< IndexType, DeviceType >;
};

#include <networks/tnlEllpackNetworkPorts_impl.h>


#endif	/* TNLELLPACKNETWORKPORTS_H */

