/***************************************************************************
                          tnlVectorOperations_impl.cu  -  description
                             -------------------
    begin                : Jul 20, 2013
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

#include <core/vectors/tnlVectorOperations.h> 

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

/****
 * Max
 */
template int         tnlVectorOperations< tnlCuda >::getVectorMax( const tnlVector< int, tnlCuda, int >& v );
template long int    tnlVectorOperations< tnlCuda >::getVectorMax( const tnlVector< long int, tnlCuda, int >& v );
template float       tnlVectorOperations< tnlCuda >::getVectorMax( const tnlVector< float, tnlCuda, int >& v );
template double      tnlVectorOperations< tnlCuda >::getVectorMax( const tnlVector< double, tnlCuda, int >& v );
//template long double tnlVectorOperations< tnlCuda >::getVectorMax( const tnlVector< long double, tnlCuda, int >& v );
template int         tnlVectorOperations< tnlCuda >::getVectorMax( const tnlVector< int, tnlCuda, long int >& v );
template long int    tnlVectorOperations< tnlCuda >::getVectorMax( const tnlVector< long int, tnlCuda, long int >& v );
template float       tnlVectorOperations< tnlCuda >::getVectorMax( const tnlVector< float, tnlCuda, long int >& v );
template double      tnlVectorOperations< tnlCuda >::getVectorMax( const tnlVector< double, tnlCuda, long int >& v );
//template long double tnlVectorOperations< tnlCuda >::getVectorMax( const tnlVector< long double, tnlCuda, long int >& v );

/****
 * Min
 */
template int         tnlVectorOperations< tnlCuda >::getVectorMin( const tnlVector< int, tnlCuda, int >& v );
template long int    tnlVectorOperations< tnlCuda >::getVectorMin( const tnlVector< long int, tnlCuda, int >& v );
template float       tnlVectorOperations< tnlCuda >::getVectorMin( const tnlVector< float, tnlCuda, int >& v );
template double      tnlVectorOperations< tnlCuda >::getVectorMin( const tnlVector< double, tnlCuda, int >& v );
//template long double tnlVectorOperations< tnlCuda >::getVectorMin( const tnlVector< long double, tnlCuda, int >& v );
template int         tnlVectorOperations< tnlCuda >::getVectorMin( const tnlVector< int, tnlCuda, long int >& v );
template long int    tnlVectorOperations< tnlCuda >::getVectorMin( const tnlVector< long int, tnlCuda, long int >& v );
template float       tnlVectorOperations< tnlCuda >::getVectorMin( const tnlVector< float, tnlCuda, long int >& v );
template double      tnlVectorOperations< tnlCuda >::getVectorMin( const tnlVector< double, tnlCuda, long int >& v );
//template long double tnlVectorOperations< tnlCuda >::getVectorMin( const tnlVector< long double, tnlCuda, long int >& v );

/****
 * Abs max
 */
template int         tnlVectorOperations< tnlCuda >::getVectorAbsMax( const tnlVector< int, tnlCuda, int >& v );
template long int    tnlVectorOperations< tnlCuda >::getVectorAbsMax( const tnlVector< long int, tnlCuda, int >& v );
template float       tnlVectorOperations< tnlCuda >::getVectorAbsMax( const tnlVector< float, tnlCuda, int >& v );
template double      tnlVectorOperations< tnlCuda >::getVectorAbsMax( const tnlVector< double, tnlCuda, int >& v );
//template long double tnlVectorOperations< tnlCuda >::getVectorAbsMax( const tnlVector< long double, tnlCuda, int >& v );
template int         tnlVectorOperations< tnlCuda >::getVectorAbsMax( const tnlVector< int, tnlCuda, long int >& v );
template long int    tnlVectorOperations< tnlCuda >::getVectorAbsMax( const tnlVector< long int, tnlCuda, long int >& v );
template float       tnlVectorOperations< tnlCuda >::getVectorAbsMax( const tnlVector< float, tnlCuda, long int >& v );
template double      tnlVectorOperations< tnlCuda >::getVectorAbsMax( const tnlVector< double, tnlCuda, long int >& v );
//template long double tnlVectorOperations< tnlCuda >::getVectorAbsMax( const tnlVector< long double, tnlCuda, long int >& v );

/****
 * Abs min
 */
template int         tnlVectorOperations< tnlCuda >::getVectorAbsMin( const tnlVector< int, tnlCuda, int >& v );
template long int    tnlVectorOperations< tnlCuda >::getVectorAbsMin( const tnlVector< long int, tnlCuda, int >& v );
template float       tnlVectorOperations< tnlCuda >::getVectorAbsMin( const tnlVector< float, tnlCuda, int >& v );
template double      tnlVectorOperations< tnlCuda >::getVectorAbsMin( const tnlVector< double, tnlCuda, int >& v );
//template long double tnlVectorOperations< tnlCuda >::getVectorAbsMin( const tnlVector< long double, tnlCuda, int >& v );
template int         tnlVectorOperations< tnlCuda >::getVectorAbsMin( const tnlVector< int, tnlCuda, long int >& v );
template long int    tnlVectorOperations< tnlCuda >::getVectorAbsMin( const tnlVector< long int, tnlCuda, long int >& v );
template float       tnlVectorOperations< tnlCuda >::getVectorAbsMin( const tnlVector< float, tnlCuda, long int >& v );
template double      tnlVectorOperations< tnlCuda >::getVectorAbsMin( const tnlVector< double, tnlCuda, long int >& v );
//template long double tnlVectorOperations< tnlCuda >::getVectorAbsMin( const tnlVector< long double, tnlCuda, long int >& v );

/****
 * Lp norm
 */
template int         tnlVectorOperations< tnlCuda >::getVectorLpNorm( const tnlVector< int, tnlCuda, int >& v, const int& p );
template long int    tnlVectorOperations< tnlCuda >::getVectorLpNorm( const tnlVector< long int, tnlCuda, int >& v, const long int& p );
template float       tnlVectorOperations< tnlCuda >::getVectorLpNorm( const tnlVector< float, tnlCuda, int >& v, const float& p );
template double      tnlVectorOperations< tnlCuda >::getVectorLpNorm( const tnlVector< double, tnlCuda, int >& v, const double& p );
//template long double tnlVectorOperations< tnlCuda >::getVectorLpNorm( const tnlVector< long double, tnlCuda, int >& v, const long double& p );
template int         tnlVectorOperations< tnlCuda >::getVectorLpNorm( const tnlVector< int, tnlCuda, long int >& v, const int& p );
template long int    tnlVectorOperations< tnlCuda >::getVectorLpNorm( const tnlVector< long int, tnlCuda, long int >& v, const long int& p );
template float       tnlVectorOperations< tnlCuda >::getVectorLpNorm( const tnlVector< float, tnlCuda, long int >& v, const float& p );
template double      tnlVectorOperations< tnlCuda >::getVectorLpNorm( const tnlVector< double, tnlCuda, long int >& v, const double& p );
//template long double tnlVectorOperations< tnlCuda >::getVectorLpNorm( const tnlVector< long double, tnlCuda, long int >& v, const long double& p );

/****
 * Sum
 */
template int         tnlVectorOperations< tnlCuda >::getVectorSum( const tnlVector< int, tnlCuda, int >& v );
template long int    tnlVectorOperations< tnlCuda >::getVectorSum( const tnlVector< long int, tnlCuda, int >& v );
template float       tnlVectorOperations< tnlCuda >::getVectorSum( const tnlVector< float, tnlCuda, int >& v );
template double      tnlVectorOperations< tnlCuda >::getVectorSum( const tnlVector< double, tnlCuda, int >& v );
//template long double tnlVectorOperations< tnlCuda >::getVectorSum( const tnlVector< long double, tnlCuda, int >& v );
template int         tnlVectorOperations< tnlCuda >::getVectorSum( const tnlVector< int, tnlCuda, long int >& v );
template long int    tnlVectorOperations< tnlCuda >::getVectorSum( const tnlVector< long int, tnlCuda, long int >& v );
template float       tnlVectorOperations< tnlCuda >::getVectorSum( const tnlVector< float, tnlCuda, long int >& v );
template double      tnlVectorOperations< tnlCuda >::getVectorSum( const tnlVector< double, tnlCuda, long int >& v );
//template long double tnlVectorOperations< tnlCuda >::getVectorSum( const tnlVector< long double, tnlCuda, long int >& v );

/****
 * Difference max
 */
template int         tnlVectorOperations< tnlCuda >::getVectorDifferenceMax( const tnlVector< int, tnlCuda, int >& v1, const tnlVector< int, tnlCuda, int >& v2 );
template long int    tnlVectorOperations< tnlCuda >::getVectorDifferenceMax( const tnlVector< long int, tnlCuda, int >& v1, const tnlVector< long int, tnlCuda, int >& v2 );
template float       tnlVectorOperations< tnlCuda >::getVectorDifferenceMax( const tnlVector< float, tnlCuda, int >& v1,  const tnlVector< float, tnlCuda, int >& v2);
template double      tnlVectorOperations< tnlCuda >::getVectorDifferenceMax( const tnlVector< double, tnlCuda, int >& v1, const tnlVector< double, tnlCuda, int >& v2 );
//template long double tnlVectorOperations< tnlCuda >::getVectorDifferenceMax( const tnlVector< long double, tnlCuda, int >& v1, const tnlVector< long double, tnlCuda, int >& v2 );
template int         tnlVectorOperations< tnlCuda >::getVectorDifferenceMax( const tnlVector< int, tnlCuda, long int >& v1, const tnlVector< int, tnlCuda, long int >& v2 );
template long int    tnlVectorOperations< tnlCuda >::getVectorDifferenceMax( const tnlVector< long int, tnlCuda, long int >& v1, const tnlVector< long int, tnlCuda, long int >& v2 );
template float       tnlVectorOperations< tnlCuda >::getVectorDifferenceMax( const tnlVector< float, tnlCuda, long int >& v1, const tnlVector< float, tnlCuda, long int >& v2 );
template double      tnlVectorOperations< tnlCuda >::getVectorDifferenceMax( const tnlVector< double, tnlCuda, long int >& v1, const tnlVector< double, tnlCuda, long int >& v2 );
//template long double tnlVectorOperations< tnlCuda >::getVectorDifferenceMax( const tnlVector< long double, tnlCuda, long int >& v1, const tnlVector< long double, tnlCuda, long int >& v2 );

/****
 * Difference min
 */
template int         tnlVectorOperations< tnlCuda >::getVectorDifferenceMin( const tnlVector< int, tnlCuda, int >& v1, const tnlVector< int, tnlCuda, int >& v2 );
template long int    tnlVectorOperations< tnlCuda >::getVectorDifferenceMin( const tnlVector< long int, tnlCuda, int >& v1, const tnlVector< long int, tnlCuda, int >& v2 );
template float       tnlVectorOperations< tnlCuda >::getVectorDifferenceMin( const tnlVector< float, tnlCuda, int >& v1,  const tnlVector< float, tnlCuda, int >& v2);
template double      tnlVectorOperations< tnlCuda >::getVectorDifferenceMin( const tnlVector< double, tnlCuda, int >& v1, const tnlVector< double, tnlCuda, int >& v2 );
//template long double tnlVectorOperations< tnlCuda >::getVectorDifferenceMin( const tnlVector< long double, tnlCuda, int >& v1, const tnlVector< long double, tnlCuda, int >& v2 );
template int         tnlVectorOperations< tnlCuda >::getVectorDifferenceMin( const tnlVector< int, tnlCuda, long int >& v1, const tnlVector< int, tnlCuda, long int >& v2 );
template long int    tnlVectorOperations< tnlCuda >::getVectorDifferenceMin( const tnlVector< long int, tnlCuda, long int >& v1, const tnlVector< long int, tnlCuda, long int >& v2 );
template float       tnlVectorOperations< tnlCuda >::getVectorDifferenceMin( const tnlVector< float, tnlCuda, long int >& v1, const tnlVector< float, tnlCuda, long int >& v2 );
template double      tnlVectorOperations< tnlCuda >::getVectorDifferenceMin( const tnlVector< double, tnlCuda, long int >& v1, const tnlVector< double, tnlCuda, long int >& v2 );
//template long double tnlVectorOperations< tnlCuda >::getVectorDifferenceMin( const tnlVector< long double, tnlCuda, long int >& v1, const tnlVector< long double, tnlCuda, long int >& v2 );

/****
 * Difference abs max
 */
template int         tnlVectorOperations< tnlCuda >::getVectorDifferenceAbsMax( const tnlVector< int, tnlCuda, int >& v1, const tnlVector< int, tnlCuda, int >& v2 );
template long int    tnlVectorOperations< tnlCuda >::getVectorDifferenceAbsMax( const tnlVector< long int, tnlCuda, int >& v1, const tnlVector< long int, tnlCuda, int >& v2 );
template float       tnlVectorOperations< tnlCuda >::getVectorDifferenceAbsMax( const tnlVector< float, tnlCuda, int >& v1,  const tnlVector< float, tnlCuda, int >& v2);
template double      tnlVectorOperations< tnlCuda >::getVectorDifferenceAbsMax( const tnlVector< double, tnlCuda, int >& v1, const tnlVector< double, tnlCuda, int >& v2 );
//template long double tnlVectorOperations< tnlCuda >::getVectorDifferenceAbsMax( const tnlVector< long double, tnlCuda, int >& v1, const tnlVector< long double, tnlCuda, int >& v2 );
template int         tnlVectorOperations< tnlCuda >::getVectorDifferenceAbsMax( const tnlVector< int, tnlCuda, long int >& v1, const tnlVector< int, tnlCuda, long int >& v2 );
template long int    tnlVectorOperations< tnlCuda >::getVectorDifferenceAbsMax( const tnlVector< long int, tnlCuda, long int >& v1, const tnlVector< long int, tnlCuda, long int >& v2 );
template float       tnlVectorOperations< tnlCuda >::getVectorDifferenceAbsMax( const tnlVector< float, tnlCuda, long int >& v1, const tnlVector< float, tnlCuda, long int >& v2 );
template double      tnlVectorOperations< tnlCuda >::getVectorDifferenceAbsMax( const tnlVector< double, tnlCuda, long int >& v1, const tnlVector< double, tnlCuda, long int >& v2 );
//template long double tnlVectorOperations< tnlCuda >::getVectorDifferenceAbsMax( const tnlVector< long double, tnlCuda, long int >& v1, const tnlVector< long double, tnlCuda, long int >& v2 );

/****
 * Difference abs min
 */
template int         tnlVectorOperations< tnlCuda >::getVectorDifferenceAbsMin( const tnlVector< int, tnlCuda, int >& v1, const tnlVector< int, tnlCuda, int >& v2 );
template long int    tnlVectorOperations< tnlCuda >::getVectorDifferenceAbsMin( const tnlVector< long int, tnlCuda, int >& v1, const tnlVector< long int, tnlCuda, int >& v2 );
template float       tnlVectorOperations< tnlCuda >::getVectorDifferenceAbsMin( const tnlVector< float, tnlCuda, int >& v1,  const tnlVector< float, tnlCuda, int >& v2);
template double      tnlVectorOperations< tnlCuda >::getVectorDifferenceAbsMin( const tnlVector< double, tnlCuda, int >& v1, const tnlVector< double, tnlCuda, int >& v2 );
//template long double tnlVectorOperations< tnlCuda >::getVectorDifferenceAbsMin( const tnlVector< long double, tnlCuda, int >& v1, const tnlVector< long double, tnlCuda, int >& v2 );
template int         tnlVectorOperations< tnlCuda >::getVectorDifferenceAbsMin( const tnlVector< int, tnlCuda, long int >& v1, const tnlVector< int, tnlCuda, long int >& v2 );
template long int    tnlVectorOperations< tnlCuda >::getVectorDifferenceAbsMin( const tnlVector< long int, tnlCuda, long int >& v1, const tnlVector< long int, tnlCuda, long int >& v2 );
template float       tnlVectorOperations< tnlCuda >::getVectorDifferenceAbsMin( const tnlVector< float, tnlCuda, long int >& v1, const tnlVector< float, tnlCuda, long int >& v2 );
template double      tnlVectorOperations< tnlCuda >::getVectorDifferenceAbsMin( const tnlVector< double, tnlCuda, long int >& v1, const tnlVector< double, tnlCuda, long int >& v2 );
//template long double tnlVectorOperations< tnlCuda >::getVectorDifferenceAbsMin( const tnlVector< long double, tnlCuda, long int >& v1, const tnlVector< long double, tnlCuda, long int >& v2 );
        
#endif
 