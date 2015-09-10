/***************************************************************************
                          tnlEllpackGraph.h  -  description
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

#ifndef TNLELLPACKGRAPH_H
#define	TNLELLPACKGRAPH_H

#include <graphs/sparse/tnlEllpackGraphLinksAccessor.h>

template< typename Device = tnlHost,
          typename Index = int >
class tnlEllpackGraphConstLinksAccessor;

template< typename Device = tnlHost,
          typename Index = int >
class tnlEllpackGraph
{
   public:
      
      typedef Device                                                            DeviceType;
      typedef Index                                                             IndexType;
      typedef tnlEllpackGraphLinksAccessor< DeviceType, IndexType >       LinksAccessorType;
      typedef tnlEllpackGraphConstLinksAccessor< DeviceType, IndexType >  ConstLinksAccessorType;
      typedef tnlVector< IndexType, DeviceType, IndexType >                     LinksPerNodesVectorType;
            
      tnlEllpackGraph();
      
      void setNumberOfNodes( const IndexType nodes );
      
      void setNumberOfLinksPerNode( const LinksPerNodesVectorType& linksPerNode );
      
      LinksAccessorType getNodeLinksAccessor( const IndexType& nodeIndex );
      
      ConstLinksAccessorType getNodeLinksAccessor( const IndexType& nodeIndex ) const;
      
   protected:
      
      tnlVector< IndexType, DeviceType, IndexType > links;
      
      IndexType maxLinksPerNode, numberOfNodes;
};


#endif	/* TNLELLPACKGRAPH_H */

