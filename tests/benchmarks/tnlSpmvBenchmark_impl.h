/***************************************************************************
                          tnlSpmvBenchmark_impl.h  -  description
                             -------------------
    begin                : Dec 29, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
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

#ifndef TNLSPMVBENCHMARK_IMPL_H_
#define TNLSPMVBENCHMARK_IMPL_H_

template< typename Real,
          typename Device,
          typename Index >
bool tnlSpmvBenchmark< tnlCSRMatrix< Real, Device, Index > >::setup( const tnlCSRMatrix< RealType, tnlHost, IndexType >& matrix )
{
   this->matrix = matrix;
}

template< typename Real,
          typename Device,
          typename Index >
void tnlSpmvBenchmark< tnlCSRMatrix< Real, Device, Index > >::tearDown()
{
   this->matrix.reset();
}

template< typename Real,
          typename Device,
          typename Index >
void tnlSpmvBenchmark< tnlCSRMatrix< Real, Device, Index > >::writeProgress() const
{
}

#endif /* TNLSPMVBENCHMARK_IMPL_H_ */
