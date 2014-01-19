/***************************************************************************
                          cuda-prefix-sum.h  -  description
                             -------------------
    begin                : Jan 18, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
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

#ifndef CUDA_PREFIX_SUM_H_
#define CUDA_PREFIX_SUM_H_

enum enumPrefixSumType { exclusivePrefixSum = 0,
                         inclusivePrefixSum };

template< typename DataType,
          template< typename T > class Operation,
          typename Index >
bool cudaPrefixSum( const Index size,
                    const Index blockSize,
                    const DataType *deviceInput,
                    DataType* deviceOutput,
                    const enumPrefixSumType prefixSumType = inclusivePrefixSum );


#endif /* CUDA_PREFIX_SUM_H_ */
