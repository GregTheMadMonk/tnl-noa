/***************************************************************************
                          tnlEllpackGraph_impl.h  -  description
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

#ifndef TNLELLPACKGRAPH_IMPL_H
#define	TNLELLPACKGRAPH_IMPL_H

#include "tnlEllpackGraph.h"


template< typename Device,
          typename Index >
tnlEllpackGraph< Device, Index >::
tnlEllpackGraph()
:  maxLinksPerNode( 0 ), numberOfNodes( 0 )
{
}

template< typename Device,
          typename Index >
void 
tnlEllpackGraph< Device, Index >::
setNumberOfNodes( const IndexType nodes )
{
   this->numberOfNodes = nodes;
}

template< typename Device,
          typename Index >
void
tnlEllpackGraph< Device, Index >::
setNumberOfLinksPerNode( const LinksPerNodesVectorType& linksPerNode )
{
   tnlAssert( linksPerNode.getSize() == this->numberOfNodes,
              cerr << "linksPerNode.getSize() =  " << linksPerNode.getSize()
                   << "this->numberOfNodes = " << this->numberOfNodes );
   this->maxLinksPerNode = linksPerNode.max();
   
   tnlAssert( this->maxLinksPerNode >= 0, 
              cerr << "this->maxLinksPerNode = " << this->maxLinksPerNode );
}

template< typename Device,
          typename Index >
typename tnlEllpackGraph< Device, Index >::LinksAccessorType 
tnlEllpackGraph< Device, Index >::
getNodeLinksAccessor( const IndexType& nodeIndex )
{
   return LinksAccessorType( this->links.getData(), nodeIndex, this->maxLinksPerNode );
}

template< typename Device,
          typename Index >
typename tnlEllpackGraph< Device, Index >::ConstLinksAccessorType
tnlEllpackGraph< Device, Index >::
getNodeLinksAccessor( const IndexType& nodeIndex ) const
{
   return ConstLinksAccessorType( this->links.getData(), nodeIndex, this->maxLinksPerNode );
}




#endif	/* TNLELLPACKGRAPH_IMPL_H */

