/***************************************************************************
                          tnlEllpackGraph.h  -  description
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

#ifndef TNLELLPACKGRAPHLINKSACCESSOR_H
#define	TNLELLPACKGRAPHLINKSACCESSOR_H

template< typename Device,
          typename Index >
class tnlEllpackGraphLinksAccessor
{
   public:
      
      typedef Device                                         DeviceType;
      typedef Index                                          IndexType;
      typedef tnlEllpackGraph< DeviceType, IndexType > GraphType;
      
      void setLink( const IndexType linkIndex,
                    const IndexType targetNode );
      
      IndexType getLinkTarget( const IndexType linkIndex ) const;
      
      IndexType& operator[]( const IndexType linkIndex );
      
      const IndexType& operator[]( const IndexType linkIndex ) const;
      
   protected:
      
      tnlEllpackGraphLinksAccessor( IndexType* graphLinks, 
                                          const IndexType node,
                                          const maxLinksPerNode );
      
      IndexType* links;
      
      IndexType step, maxLinksPerNode;
      
      friend tnlEllpackGraph< IndexType, DeviceType >;
};

#include <graphs/sparse/tnlEllpackGraphLinksAccessor_impl.h>


#endif	/* TNLELLPACKGRAPHLINKSACCESSOR_H */

