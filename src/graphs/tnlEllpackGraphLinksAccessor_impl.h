/***************************************************************************
                          tnlEllpackGraph_impl.h  -  description
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

#ifndef TNLELLPACKGRAPHLINKSACCESSOR_IMPL_H
#define	TNLELLPACKGRAPHLINKSACCESSOR_IMPL_H

template< typename Device,
          typename Index >
tnlEllpackGraphLinksAccessor< Device, Index >::
tnlEllpackGraphLinksAccessor( IndexType* graphLinks, 
                                    const IndexType node,
                                    const IndexType maxLinksPerNode )
{
   this->links = &graphLinks[ node * maxLinksPerNode ];
   this->maxLinksPerNode = maxLinksPerNode;
}


template< typename Device,
          typename Index >
void 
tnlEllpackGraphLinksAccessor< Device, Index >::
setLink( const IndexType linkIndex,
         const IndexType targetNode )
{
   links[ linkIndex ] = targetNode;
}

template< typename Device,
          typename Index >
Index
tnlEllpackGraphLinksAccessor< Device, Index >::
getLinkTarget( const IndexType linkIndex ) const
{
   return links[ linkIndex ];
}

template< typename Device,
          typename Index >
Index&
tnlEllpackGraphLinksAccessor< Device, Index >::
operator[]( const IndexType linkIndex )
{
   return links[ linkIndex ];
}

template< typename Device,
          typename Index >
const Index&
tnlEllpackGraphLinksAccessor< Device, Index >::
operator[]( const IndexType linkIndex ) const
{
   return links[ linkIndex ];
}


#endif	/* TNLELLPACKGRAPHLINKSACCESSOR_IMPL_H */

