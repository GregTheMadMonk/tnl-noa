/***************************************************************************
                          tnlVectorOperationsCuda_impl.cu  -  description
                             -------------------
    begin                : Jul 20, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <core/vectors/tnlVectorOperations.h>

namespace TNL {

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

/****
 * Max
 */
template int         tnlVectorOperations< tnlCuda >::getVectorMax( const tnlVector< int, tnlCuda, int >& v );
template long int    tnlVectorOperations< tnlCuda >::getVectorMax( const tnlVector< long int, tnlCuda, int >& v );
#ifdef INSTANTIATE_FLOAT
template float       tnlVectorOperations< tnlCuda >::getVectorMax( const tnlVector< float, tnlCuda, int >& v );
#endif
template double      tnlVectorOperations< tnlCuda >::getVectorMax( const tnlVector< double, tnlCuda, int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double tnlVectorOperations< tnlCuda >::getVectorMax( const tnlVector< long double, tnlCuda, int >& v );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         tnlVectorOperations< tnlCuda >::getVectorMax( const tnlVector< int, tnlCuda, long int >& v );
template long int    tnlVectorOperations< tnlCuda >::getVectorMax( const tnlVector< long int, tnlCuda, long int >& v );
#ifdef INSTANTIATE_FLOAT
template float       tnlVectorOperations< tnlCuda >::getVectorMax( const tnlVector< float, tnlCuda, long int >& v );
#endif
template double      tnlVectorOperations< tnlCuda >::getVectorMax( const tnlVector< double, tnlCuda, long int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double tnlVectorOperations< tnlCuda >::getVectorMax( const tnlVector< long double, tnlCuda, long int >& v );
#endif
#endif

/****
 * Min
 */
template int         tnlVectorOperations< tnlCuda >::getVectorMin( const tnlVector< int, tnlCuda, int >& v );
template long int    tnlVectorOperations< tnlCuda >::getVectorMin( const tnlVector< long int, tnlCuda, int >& v );
#ifdef INSTANTIATE_FLOAT
template float       tnlVectorOperations< tnlCuda >::getVectorMin( const tnlVector< float, tnlCuda, int >& v );
#endif
template double      tnlVectorOperations< tnlCuda >::getVectorMin( const tnlVector< double, tnlCuda, int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double tnlVectorOperations< tnlCuda >::getVectorMin( const tnlVector< long double, tnlCuda, int >& v );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         tnlVectorOperations< tnlCuda >::getVectorMin( const tnlVector< int, tnlCuda, long int >& v );
template long int    tnlVectorOperations< tnlCuda >::getVectorMin( const tnlVector< long int, tnlCuda, long int >& v );
#ifdef INSTANTIATE_FLOAT
template float       tnlVectorOperations< tnlCuda >::getVectorMin( const tnlVector< float, tnlCuda, long int >& v );
#endif
template double      tnlVectorOperations< tnlCuda >::getVectorMin( const tnlVector< double, tnlCuda, long int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double tnlVectorOperations< tnlCuda >::getVectorMin( const tnlVector< long double, tnlCuda, long int >& v );
#endif
#endif

/****
 * Abs max
 */
template int         tnlVectorOperations< tnlCuda >::getVectorAbsMax( const tnlVector< int, tnlCuda, int >& v );
template long int    tnlVectorOperations< tnlCuda >::getVectorAbsMax( const tnlVector< long int, tnlCuda, int >& v );
#ifdef INSTANTIATE_FLOAT
template float       tnlVectorOperations< tnlCuda >::getVectorAbsMax( const tnlVector< float, tnlCuda, int >& v );
#endif
template double      tnlVectorOperations< tnlCuda >::getVectorAbsMax( const tnlVector< double, tnlCuda, int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double tnlVectorOperations< tnlCuda >::getVectorAbsMax( const tnlVector< long double, tnlCuda, int >& v );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         tnlVectorOperations< tnlCuda >::getVectorAbsMax( const tnlVector< int, tnlCuda, long int >& v );
template long int    tnlVectorOperations< tnlCuda >::getVectorAbsMax( const tnlVector< long int, tnlCuda, long int >& v );
#ifdef INSTANTIATE_FLOAT
template float       tnlVectorOperations< tnlCuda >::getVectorAbsMax( const tnlVector< float, tnlCuda, long int >& v );
#endif
template double      tnlVectorOperations< tnlCuda >::getVectorAbsMax( const tnlVector< double, tnlCuda, long int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double tnlVectorOperations< tnlCuda >::getVectorAbsMax( const tnlVector< long double, tnlCuda, long int >& v );
#endif
#endif


/****
 * Abs min
 */
template int         tnlVectorOperations< tnlCuda >::getVectorAbsMin( const tnlVector< int, tnlCuda, int >& v );
template long int    tnlVectorOperations< tnlCuda >::getVectorAbsMin( const tnlVector< long int, tnlCuda, int >& v );
#ifdef INSTANTIATE_FLOAT
template float       tnlVectorOperations< tnlCuda >::getVectorAbsMin( const tnlVector< float, tnlCuda, int >& v );
#endif
template double      tnlVectorOperations< tnlCuda >::getVectorAbsMin( const tnlVector< double, tnlCuda, int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double tnlVectorOperations< tnlCuda >::getVectorAbsMin( const tnlVector< long double, tnlCuda, int >& v );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         tnlVectorOperations< tnlCuda >::getVectorAbsMin( const tnlVector< int, tnlCuda, long int >& v );
template long int    tnlVectorOperations< tnlCuda >::getVectorAbsMin( const tnlVector< long int, tnlCuda, long int >& v );
#ifdef INSTANTIATE_FLOAT
template float       tnlVectorOperations< tnlCuda >::getVectorAbsMin( const tnlVector< float, tnlCuda, long int >& v );
#endif
template double      tnlVectorOperations< tnlCuda >::getVectorAbsMin( const tnlVector< double, tnlCuda, long int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double tnlVectorOperations< tnlCuda >::getVectorAbsMin( const tnlVector< long double, tnlCuda, long int >& v );
#endif
#endif

/****
 * L2 norm
 */
template int         tnlVectorOperations< tnlCuda >::getVectorL2Norm( const tnlVector< int, tnlCuda, int >& v );
template long int    tnlVectorOperations< tnlCuda >::getVectorL2Norm( const tnlVector< long int, tnlCuda, int >& v );
#ifdef INSTANTIATE_FLOAT
template float       tnlVectorOperations< tnlCuda >::getVectorL2Norm( const tnlVector< float, tnlCuda, int >& v );
#endif
template double      tnlVectorOperations< tnlCuda >::getVectorL2Norm( const tnlVector< double, tnlCuda, int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double tnlVectorOperations< tnlCuda >::getVectorL2Norm( const tnlVector< long double, tnlCuda, int >& v );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         tnlVectorOperations< tnlCuda >::getVectorL2Norm( const tnlVector< int, tnlCuda, long int >& v );
template long int    tnlVectorOperations< tnlCuda >::getVectorL2Norm( const tnlVector< long int, tnlCuda, long int >& v );
#ifdef INSTANTIATE_FLOAT
template float       tnlVectorOperations< tnlCuda >::getVectorL2Norm( const tnlVector< float, tnlCuda, long int >& v );
#endif
template double      tnlVectorOperations< tnlCuda >::getVectorL2Norm( const tnlVector< double, tnlCuda, long int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double tnlVectorOperations< tnlCuda >::getVectorL2Norm( const tnlVector< long double, tnlCuda, long int >& v );
#endif
#endif

/****
 * L1 norm
 */
template int         tnlVectorOperations< tnlCuda >::getVectorL1Norm( const tnlVector< int, tnlCuda, int >& v );
template long int    tnlVectorOperations< tnlCuda >::getVectorL1Norm( const tnlVector< long int, tnlCuda, int >& v );
#ifdef INSTANTIATE_FLOAT
template float       tnlVectorOperations< tnlCuda >::getVectorL1Norm( const tnlVector< float, tnlCuda, int >& v );
#endif
template double      tnlVectorOperations< tnlCuda >::getVectorL1Norm( const tnlVector< double, tnlCuda, int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double tnlVectorOperations< tnlCuda >::getVectorL1Norm( const tnlVector< long double, tnlCuda, int >& v );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         tnlVectorOperations< tnlCuda >::getVectorL1Norm( const tnlVector< int, tnlCuda, long int >& v );
template long int    tnlVectorOperations< tnlCuda >::getVectorL1Norm( const tnlVector< long int, tnlCuda, long int >& v );
#ifdef INSTANTIATE_FLOAT
template float       tnlVectorOperations< tnlCuda >::getVectorL1Norm( const tnlVector< float, tnlCuda, long int >& v );
#endif
template double      tnlVectorOperations< tnlCuda >::getVectorL1Norm( const tnlVector< double, tnlCuda, long int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double tnlVectorOperations< tnlCuda >::getVectorL1Norm( const tnlVector< long double, tnlCuda, long int >& v );
#endif
#endif

/****
 * Lp norm
 */
template int         tnlVectorOperations< tnlCuda >::getVectorLpNorm( const tnlVector< int, tnlCuda, int >& v, const int& p );
template long int    tnlVectorOperations< tnlCuda >::getVectorLpNorm( const tnlVector< long int, tnlCuda, int >& v, const long int& p );
#ifdef INSTANTIATE_FLOAT
template float       tnlVectorOperations< tnlCuda >::getVectorLpNorm( const tnlVector< float, tnlCuda, int >& v, const float& p );
#endif
template double      tnlVectorOperations< tnlCuda >::getVectorLpNorm( const tnlVector< double, tnlCuda, int >& v, const double& p );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double tnlVectorOperations< tnlCuda >::getVectorLpNorm( const tnlVector< long double, tnlCuda, int >& v, const long double& p );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         tnlVectorOperations< tnlCuda >::getVectorLpNorm( const tnlVector< int, tnlCuda, long int >& v, const int& p );
template long int    tnlVectorOperations< tnlCuda >::getVectorLpNorm( const tnlVector< long int, tnlCuda, long int >& v, const long int& p );
#ifdef INSTANTIATE_FLOAT
template float       tnlVectorOperations< tnlCuda >::getVectorLpNorm( const tnlVector< float, tnlCuda, long int >& v, const float& p );
#endif
template double      tnlVectorOperations< tnlCuda >::getVectorLpNorm( const tnlVector< double, tnlCuda, long int >& v, const double& p );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double tnlVectorOperations< tnlCuda >::getVectorLpNorm( const tnlVector< long double, tnlCuda, long int >& v, const long double& p );
#endif
#endif



/****
 * Sum
 */
template int         tnlVectorOperations< tnlCuda >::getVectorSum( const tnlVector< int, tnlCuda, int >& v );
template long int    tnlVectorOperations< tnlCuda >::getVectorSum( const tnlVector< long int, tnlCuda, int >& v );
#ifdef INSTANTIATE_FLOAT
template float       tnlVectorOperations< tnlCuda >::getVectorSum( const tnlVector< float, tnlCuda, int >& v );
#endif
template double      tnlVectorOperations< tnlCuda >::getVectorSum( const tnlVector< double, tnlCuda, int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double tnlVectorOperations< tnlCuda >::getVectorSum( const tnlVector< long double, tnlCuda, int >& v );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         tnlVectorOperations< tnlCuda >::getVectorSum( const tnlVector< int, tnlCuda, long int >& v );
template long int    tnlVectorOperations< tnlCuda >::getVectorSum( const tnlVector< long int, tnlCuda, long int >& v );
#ifdef INSTANTIATE_FLOAT
template float       tnlVectorOperations< tnlCuda >::getVectorSum( const tnlVector< float, tnlCuda, long int >& v );
#endif
template double      tnlVectorOperations< tnlCuda >::getVectorSum( const tnlVector< double, tnlCuda, long int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double tnlVectorOperations< tnlCuda >::getVectorSum( const tnlVector< long double, tnlCuda, long int >& v );
#endif
#endif

/****
 * Difference max
 */
template int         tnlVectorOperations< tnlCuda >::getVectorDifferenceMax( const tnlVector< int, tnlCuda, int >& v1, const tnlVector< int, tnlCuda, int >& v2 );
template long int    tnlVectorOperations< tnlCuda >::getVectorDifferenceMax( const tnlVector< long int, tnlCuda, int >& v1, const tnlVector< long int, tnlCuda, int >& v2 );
#ifdef INSTANTIATE_FLOAT
template float       tnlVectorOperations< tnlCuda >::getVectorDifferenceMax( const tnlVector< float, tnlCuda, int >& v1,  const tnlVector< float, tnlCuda, int >& v2);
#endif
template double      tnlVectorOperations< tnlCuda >::getVectorDifferenceMax( const tnlVector< double, tnlCuda, int >& v1, const tnlVector< double, tnlCuda, int >& v2 );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double tnlVectorOperations< tnlCuda >::getVectorDifferenceMax( const tnlVector< long double, tnlCuda, int >& v1, const tnlVector< long double, tnlCuda, int >& v2 );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         tnlVectorOperations< tnlCuda >::getVectorDifferenceMax( const tnlVector< int, tnlCuda, long int >& v1, const tnlVector< int, tnlCuda, long int >& v2 );
template long int    tnlVectorOperations< tnlCuda >::getVectorDifferenceMax( const tnlVector< long int, tnlCuda, long int >& v1, const tnlVector< long int, tnlCuda, long int >& v2 );
#ifdef INSTANTIATE_FLOAT
template float       tnlVectorOperations< tnlCuda >::getVectorDifferenceMax( const tnlVector< float, tnlCuda, long int >& v1, const tnlVector< float, tnlCuda, long int >& v2 );
#endif
template double      tnlVectorOperations< tnlCuda >::getVectorDifferenceMax( const tnlVector< double, tnlCuda, long int >& v1, const tnlVector< double, tnlCuda, long int >& v2 );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double tnlVectorOperations< tnlCuda >::getVectorDifferenceMax( const tnlVector< long double, tnlCuda, long int >& v1, const tnlVector< long double, tnlCuda, long int >& v2 );
#endif
#endif

/****
 * Difference min
 */
template int         tnlVectorOperations< tnlCuda >::getVectorDifferenceMin( const tnlVector< int, tnlCuda, int >& v1, const tnlVector< int, tnlCuda, int >& v2 );
template long int    tnlVectorOperations< tnlCuda >::getVectorDifferenceMin( const tnlVector< long int, tnlCuda, int >& v1, const tnlVector< long int, tnlCuda, int >& v2 );
#ifdef INSTANTIATE_FLOAT
template float       tnlVectorOperations< tnlCuda >::getVectorDifferenceMin( const tnlVector< float, tnlCuda, int >& v1,  const tnlVector< float, tnlCuda, int >& v2);
#endif
template double      tnlVectorOperations< tnlCuda >::getVectorDifferenceMin( const tnlVector< double, tnlCuda, int >& v1, const tnlVector< double, tnlCuda, int >& v2 );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double tnlVectorOperations< tnlCuda >::getVectorDifferenceMin( const tnlVector< long double, tnlCuda, int >& v1, const tnlVector< long double, tnlCuda, int >& v2 );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         tnlVectorOperations< tnlCuda >::getVectorDifferenceMin( const tnlVector< int, tnlCuda, long int >& v1, const tnlVector< int, tnlCuda, long int >& v2 );
template long int    tnlVectorOperations< tnlCuda >::getVectorDifferenceMin( const tnlVector< long int, tnlCuda, long int >& v1, const tnlVector< long int, tnlCuda, long int >& v2 );
#ifdef INSTANTIATE_FLOAT
template float       tnlVectorOperations< tnlCuda >::getVectorDifferenceMin( const tnlVector< float, tnlCuda, long int >& v1, const tnlVector< float, tnlCuda, long int >& v2 );
#endif
template double      tnlVectorOperations< tnlCuda >::getVectorDifferenceMin( const tnlVector< double, tnlCuda, long int >& v1, const tnlVector< double, tnlCuda, long int >& v2 );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double tnlVectorOperations< tnlCuda >::getVectorDifferenceMin( const tnlVector< long double, tnlCuda, long int >& v1, const tnlVector< long double, tnlCuda, long int >& v2 );
#endif
#endif

/****
 * Difference abs max
 */
template int         tnlVectorOperations< tnlCuda >::getVectorDifferenceAbsMax( const tnlVector< int, tnlCuda, int >& v1, const tnlVector< int, tnlCuda, int >& v2 );
template long int    tnlVectorOperations< tnlCuda >::getVectorDifferenceAbsMax( const tnlVector< long int, tnlCuda, int >& v1, const tnlVector< long int, tnlCuda, int >& v2 );
#ifdef INSTANTIATE_FLOAT
template float       tnlVectorOperations< tnlCuda >::getVectorDifferenceAbsMax( const tnlVector< float, tnlCuda, int >& v1,  const tnlVector< float, tnlCuda, int >& v2);
#endif
template double      tnlVectorOperations< tnlCuda >::getVectorDifferenceAbsMax( const tnlVector< double, tnlCuda, int >& v1, const tnlVector< double, tnlCuda, int >& v2 );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double tnlVectorOperations< tnlCuda >::getVectorDifferenceAbsMax( const tnlVector< long double, tnlCuda, int >& v1, const tnlVector< long double, tnlCuda, int >& v2 );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         tnlVectorOperations< tnlCuda >::getVectorDifferenceAbsMax( const tnlVector< int, tnlCuda, long int >& v1, const tnlVector< int, tnlCuda, long int >& v2 );
template long int    tnlVectorOperations< tnlCuda >::getVectorDifferenceAbsMax( const tnlVector< long int, tnlCuda, long int >& v1, const tnlVector< long int, tnlCuda, long int >& v2 );
#ifdef INSTANTIATE_FLOAT
template float       tnlVectorOperations< tnlCuda >::getVectorDifferenceAbsMax( const tnlVector< float, tnlCuda, long int >& v1, const tnlVector< float, tnlCuda, long int >& v2 );
#endif
template double      tnlVectorOperations< tnlCuda >::getVectorDifferenceAbsMax( const tnlVector< double, tnlCuda, long int >& v1, const tnlVector< double, tnlCuda, long int >& v2 );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double tnlVectorOperations< tnlCuda >::getVectorDifferenceAbsMax( const tnlVector< long double, tnlCuda, long int >& v1, const tnlVector< long double, tnlCuda, long int >& v2 );
#endif
#endif


/****
 * Difference abs min
 */
template int         tnlVectorOperations< tnlCuda >::getVectorDifferenceAbsMin( const tnlVector< int, tnlCuda, int >& v1, const tnlVector< int, tnlCuda, int >& v2 );
template long int    tnlVectorOperations< tnlCuda >::getVectorDifferenceAbsMin( const tnlVector< long int, tnlCuda, int >& v1, const tnlVector< long int, tnlCuda, int >& v2 );
#ifdef INSTANTIATE_FLOAT
template float       tnlVectorOperations< tnlCuda >::getVectorDifferenceAbsMin( const tnlVector< float, tnlCuda, int >& v1,  const tnlVector< float, tnlCuda, int >& v2);
#endif
template double      tnlVectorOperations< tnlCuda >::getVectorDifferenceAbsMin( const tnlVector< double, tnlCuda, int >& v1, const tnlVector< double, tnlCuda, int >& v2 );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double tnlVectorOperations< tnlCuda >::getVectorDifferenceAbsMin( const tnlVector< long double, tnlCuda, int >& v1, const tnlVector< long double, tnlCuda, int >& v2 );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         tnlVectorOperations< tnlCuda >::getVectorDifferenceAbsMin( const tnlVector< int, tnlCuda, long int >& v1, const tnlVector< int, tnlCuda, long int >& v2 );
template long int    tnlVectorOperations< tnlCuda >::getVectorDifferenceAbsMin( const tnlVector< long int, tnlCuda, long int >& v1, const tnlVector< long int, tnlCuda, long int >& v2 );
#ifdef INSTANTIATE_FLOAT
template float       tnlVectorOperations< tnlCuda >::getVectorDifferenceAbsMin( const tnlVector< float, tnlCuda, long int >& v1, const tnlVector< float, tnlCuda, long int >& v2 );
#endif
template double      tnlVectorOperations< tnlCuda >::getVectorDifferenceAbsMin( const tnlVector< double, tnlCuda, long int >& v1, const tnlVector< double, tnlCuda, long int >& v2 );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double tnlVectorOperations< tnlCuda >::getVectorDifferenceAbsMin( const tnlVector< long double, tnlCuda, long int >& v1, const tnlVector< long double, tnlCuda, long int >& v2 );
#endif
#endif
 
#endif
 
} // namespace TNL