/***************************************************************************
                          tnlBackwardTimeDiscretisation.h  -  description
                             -------------------
    begin                : Apr 4, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
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


#ifndef TNLBACKWARDTIMEDISCRETISATION_H
#define	TNLBACKWARDTIMEDISCRETISATION_H

#include <core/tnlCuda.h>

class tnlBackwardTimeDiscretisation
{
    public:        
        
        template< typename RealType,
                  typename IndexType,
                  typename MatrixType >
        __cuda_callable__ static void applyTimeDiscretisation( MatrixType& matrix,
                                                               RealType& b,
                                                               const IndexType index,
                                                               const RealType& u,
                                                               const RealType& tau,
                                                               const RealType& rhs )
        {
            b += u + tau * rhs;
            matrix.addElementFast( index, index, 1.0, 1.0 );
        };
};

#endif	/* TNLBACKWARDTIMEDISCRETISATION_H */

