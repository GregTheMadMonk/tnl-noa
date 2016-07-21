/***************************************************************************
                          tnlVectorOperationsHost_impl.cpp  -  description
                             -------------------
    begin                : Jul 20, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/Vectors/VectorOperations.h>

namespace TNL {
namespace Vectors {    

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

/****
 * Max
 */
template int         tnlVectorOperations< tnlHost >::getVectorMax( const tnlVector< int, tnlHost, int >& v );
template long int    tnlVectorOperations< tnlHost >::getVectorMax( const tnlVector< long int, tnlHost, int >& v );
#ifdef INSTANTIATE_FLOAT
template float       tnlVectorOperations< tnlHost >::getVectorMax( const tnlVector< float, tnlHost, int >& v );
#endif
template double      tnlVectorOperations< tnlHost >::getVectorMax( const tnlVector< double, tnlHost, int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double tnlVectorOperations< tnlHost >::getVectorMax( const tnlVector< long double, tnlHost, int >& v );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         tnlVectorOperations< tnlHost >::getVectorMax( const tnlVector< int, tnlHost, long int >& v );
template long int    tnlVectorOperations< tnlHost >::getVectorMax( const tnlVector< long int, tnlHost, long int >& v );
#ifdef INSTANTIATE_FLOAT
template float       tnlVectorOperations< tnlHost >::getVectorMax( const tnlVector< float, tnlHost, long int >& v );
#endif
template double      tnlVectorOperations< tnlHost >::getVectorMax( const tnlVector< double, tnlHost, long int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double tnlVectorOperations< tnlHost >::getVectorMax( const tnlVector< long double, tnlHost, long int >& v );
#endif
#endif

/****
 * Min
 */
template int         tnlVectorOperations< tnlHost >::getVectorMin( const tnlVector< int, tnlHost, int >& v );
template long int    tnlVectorOperations< tnlHost >::getVectorMin( const tnlVector< long int, tnlHost, int >& v );
#ifdef INSTANTIATE_FLOAT
template float       tnlVectorOperations< tnlHost >::getVectorMin( const tnlVector< float, tnlHost, int >& v );
#endif
template double      tnlVectorOperations< tnlHost >::getVectorMin( const tnlVector< double, tnlHost, int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double tnlVectorOperations< tnlHost >::getVectorMin( const tnlVector< long double, tnlHost, int >& v );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         tnlVectorOperations< tnlHost >::getVectorMin( const tnlVector< int, tnlHost, long int >& v );
template long int    tnlVectorOperations< tnlHost >::getVectorMin( const tnlVector< long int, tnlHost, long int >& v );
#ifdef INSTANTIATE_FLOAT
template float       tnlVectorOperations< tnlHost >::getVectorMin( const tnlVector< float, tnlHost, long int >& v );
#endif
template double      tnlVectorOperations< tnlHost >::getVectorMin( const tnlVector< double, tnlHost, long int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double tnlVectorOperations< tnlHost >::getVectorMin( const tnlVector< long double, tnlHost, long int >& v );
#endif
#endif

/****
 * Abs max
 */
template int         tnlVectorOperations< tnlHost >::getVectorAbsMax( const tnlVector< int, tnlHost, int >& v );
template long int    tnlVectorOperations< tnlHost >::getVectorAbsMax( const tnlVector< long int, tnlHost, int >& v );
#ifdef INSTANTIATE_FLOAT
template float       tnlVectorOperations< tnlHost >::getVectorAbsMax( const tnlVector< float, tnlHost, int >& v );
#endif
template double      tnlVectorOperations< tnlHost >::getVectorAbsMax( const tnlVector< double, tnlHost, int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double tnlVectorOperations< tnlHost >::getVectorAbsMax( const tnlVector< long double, tnlHost, int >& v );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         tnlVectorOperations< tnlHost >::getVectorAbsMax( const tnlVector< int, tnlHost, long int >& v );
template long int    tnlVectorOperations< tnlHost >::getVectorAbsMax( const tnlVector< long int, tnlHost, long int >& v );
#ifdef INSTANTIATE_FLOAT
template float       tnlVectorOperations< tnlHost >::getVectorAbsMax( const tnlVector< float, tnlHost, long int >& v );
#endif
template double      tnlVectorOperations< tnlHost >::getVectorAbsMax( const tnlVector< double, tnlHost, long int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double tnlVectorOperations< tnlHost >::getVectorAbsMax( const tnlVector< long double, tnlHost, long int >& v );
#endif
#endif

/****
 * Abs min
 */
template int         tnlVectorOperations< tnlHost >::getVectorAbsMin( const tnlVector< int, tnlHost, int >& v );
template long int    tnlVectorOperations< tnlHost >::getVectorAbsMin( const tnlVector< long int, tnlHost, int >& v );
#ifdef INSTANTIATE_FLOAT
template float       tnlVectorOperations< tnlHost >::getVectorAbsMin( const tnlVector< float, tnlHost, int >& v );
#endif
template double      tnlVectorOperations< tnlHost >::getVectorAbsMin( const tnlVector< double, tnlHost, int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double tnlVectorOperations< tnlHost >::getVectorAbsMin( const tnlVector< long double, tnlHost, int >& v );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         tnlVectorOperations< tnlHost >::getVectorAbsMin( const tnlVector< int, tnlHost, long int >& v );
template long int    tnlVectorOperations< tnlHost >::getVectorAbsMin( const tnlVector< long int, tnlHost, long int >& v );
#ifdef INSTANTIATE_FLOAT
template float       tnlVectorOperations< tnlHost >::getVectorAbsMin( const tnlVector< float, tnlHost, long int >& v );
#endif
template double      tnlVectorOperations< tnlHost >::getVectorAbsMin( const tnlVector< double, tnlHost, long int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double tnlVectorOperations< tnlHost >::getVectorAbsMin( const tnlVector< long double, tnlHost, long int >& v );
#endif
#endif

/****
 * L1 norm
 */
template int         tnlVectorOperations< tnlHost >::getVectorL1Norm( const tnlVector< int, tnlHost, int >& v );
template long int    tnlVectorOperations< tnlHost >::getVectorL1Norm( const tnlVector< long int, tnlHost, int >& v );
#ifdef INSTANTIATE_FLOAT
template float       tnlVectorOperations< tnlHost >::getVectorL1Norm( const tnlVector< float, tnlHost, int >& v );
#endif
template double      tnlVectorOperations< tnlHost >::getVectorL1Norm( const tnlVector< double, tnlHost, int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double tnlVectorOperations< tnlHost >::getVectorL1Norm( const tnlVector< long double, tnlHost, int >& v );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         tnlVectorOperations< tnlHost >::getVectorL1Norm( const tnlVector< int, tnlHost, long int >& v );
template long int    tnlVectorOperations< tnlHost >::getVectorL1Norm( const tnlVector< long int, tnlHost, long int >& v );
#ifdef INSTANTIATE_FLOAT
template float       tnlVectorOperations< tnlHost >::getVectorL1Norm( const tnlVector< float, tnlHost, long int >& v );
#endif
template double      tnlVectorOperations< tnlHost >::getVectorL1Norm( const tnlVector< double, tnlHost, long int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double tnlVectorOperations< tnlHost >::getVectorL1Norm( const tnlVector< long double, tnlHost, long int >& v );
#endif
#endif

/****
 * L2 norm
 */
template int         tnlVectorOperations< tnlHost >::getVectorL2Norm( const tnlVector< int, tnlHost, int >& v );
template long int    tnlVectorOperations< tnlHost >::getVectorL2Norm( const tnlVector< long int, tnlHost, int >& v );
#ifdef INSTANTIATE_FLOAT
template float       tnlVectorOperations< tnlHost >::getVectorL2Norm( const tnlVector< float, tnlHost, int >& v );
#endif
template double      tnlVectorOperations< tnlHost >::getVectorL2Norm( const tnlVector< double, tnlHost, int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double tnlVectorOperations< tnlHost >::getVectorL2Norm( const tnlVector< long double, tnlHost, int >& v );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         tnlVectorOperations< tnlHost >::getVectorL2Norm( const tnlVector< int, tnlHost, long int >& v );
template long int    tnlVectorOperations< tnlHost >::getVectorL2Norm( const tnlVector< long int, tnlHost, long int >& v );
#ifdef INSTANTIATE_FLOAT
template float       tnlVectorOperations< tnlHost >::getVectorL2Norm( const tnlVector< float, tnlHost, long int >& v );
#endif
template double      tnlVectorOperations< tnlHost >::getVectorL2Norm( const tnlVector< double, tnlHost, long int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double tnlVectorOperations< tnlHost >::getVectorL2Norm( const tnlVector< long double, tnlHost, long int >& v );
#endif
#endif


/****
 * Lp norm
 */
template int         tnlVectorOperations< tnlHost >::getVectorLpNorm( const tnlVector< int, tnlHost, int >& v, const int& p );
template long int    tnlVectorOperations< tnlHost >::getVectorLpNorm( const tnlVector< long int, tnlHost, int >& v, const long int& p );
#ifdef INSTANTIATE_FLOAT
template float       tnlVectorOperations< tnlHost >::getVectorLpNorm( const tnlVector< float, tnlHost, int >& v, const float& p );
#endif
template double      tnlVectorOperations< tnlHost >::getVectorLpNorm( const tnlVector< double, tnlHost, int >& v, const double& p );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double tnlVectorOperations< tnlHost >::getVectorLpNorm( const tnlVector< long double, tnlHost, int >& v, const long double& p );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         tnlVectorOperations< tnlHost >::getVectorLpNorm( const tnlVector< int, tnlHost, long int >& v, const int& p );
template long int    tnlVectorOperations< tnlHost >::getVectorLpNorm( const tnlVector< long int, tnlHost, long int >& v, const long int& p );
#ifdef INSTANTIATE_FLOAT
template float       tnlVectorOperations< tnlHost >::getVectorLpNorm( const tnlVector< float, tnlHost, long int >& v, const float& p );
#endif
template double      tnlVectorOperations< tnlHost >::getVectorLpNorm( const tnlVector< double, tnlHost, long int >& v, const double& p );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double tnlVectorOperations< tnlHost >::getVectorLpNorm( const tnlVector< long double, tnlHost, long int >& v, const long double& p );
#endif
#endif



/****
 * Sum
 */
template int         tnlVectorOperations< tnlHost >::getVectorSum( const tnlVector< int, tnlHost, int >& v );
template long int    tnlVectorOperations< tnlHost >::getVectorSum( const tnlVector< long int, tnlHost, int >& v );
#ifdef INSTANTIATE_FLOAT
template float       tnlVectorOperations< tnlHost >::getVectorSum( const tnlVector< float, tnlHost, int >& v );
#endif
template double      tnlVectorOperations< tnlHost >::getVectorSum( const tnlVector< double, tnlHost, int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double tnlVectorOperations< tnlHost >::getVectorSum( const tnlVector< long double, tnlHost, int >& v );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         tnlVectorOperations< tnlHost >::getVectorSum( const tnlVector< int, tnlHost, long int >& v );
template long int    tnlVectorOperations< tnlHost >::getVectorSum( const tnlVector< long int, tnlHost, long int >& v );
#ifdef INSTANTIATE_FLOAT
template float       tnlVectorOperations< tnlHost >::getVectorSum( const tnlVector< float, tnlHost, long int >& v );
#endif
template double      tnlVectorOperations< tnlHost >::getVectorSum( const tnlVector< double, tnlHost, long int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double tnlVectorOperations< tnlHost >::getVectorSum( const tnlVector< long double, tnlHost, long int >& v );
#endif
#endif

/****
 * Difference max
 */
template int         tnlVectorOperations< tnlHost >::getVectorDifferenceMax( const tnlVector< int, tnlHost, int >& v1, const tnlVector< int, tnlHost, int >& v2 );
template long int    tnlVectorOperations< tnlHost >::getVectorDifferenceMax( const tnlVector< long int, tnlHost, int >& v1, const tnlVector< long int, tnlHost, int >& v2 );
#ifdef INSTANTIATE_FLOAT
template float       tnlVectorOperations< tnlHost >::getVectorDifferenceMax( const tnlVector< float, tnlHost, int >& v1,  const tnlVector< float, tnlHost, int >& v2);
#endif
template double      tnlVectorOperations< tnlHost >::getVectorDifferenceMax( const tnlVector< double, tnlHost, int >& v1, const tnlVector< double, tnlHost, int >& v2 );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double tnlVectorOperations< tnlHost >::getVectorDifferenceMax( const tnlVector< long double, tnlHost, int >& v1, const tnlVector< long double, tnlHost, int >& v2 );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         tnlVectorOperations< tnlHost >::getVectorDifferenceMax( const tnlVector< int, tnlHost, long int >& v1, const tnlVector< int, tnlHost, long int >& v2 );
template long int    tnlVectorOperations< tnlHost >::getVectorDifferenceMax( const tnlVector< long int, tnlHost, long int >& v1, const tnlVector< long int, tnlHost, long int >& v2 );
#ifdef INSTANTIATE_FLOAT
template float       tnlVectorOperations< tnlHost >::getVectorDifferenceMax( const tnlVector< float, tnlHost, long int >& v1, const tnlVector< float, tnlHost, long int >& v2 );
#endif
template double      tnlVectorOperations< tnlHost >::getVectorDifferenceMax( const tnlVector< double, tnlHost, long int >& v1, const tnlVector< double, tnlHost, long int >& v2 );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double tnlVectorOperations< tnlHost >::getVectorDifferenceMax( const tnlVector< long double, tnlHost, long int >& v1, const tnlVector< long double, tnlHost, long int >& v2 );
#endif
#endif

/****
 * Difference min
 */
template int         tnlVectorOperations< tnlHost >::getVectorDifferenceMin( const tnlVector< int, tnlHost, int >& v1, const tnlVector< int, tnlHost, int >& v2 );
template long int    tnlVectorOperations< tnlHost >::getVectorDifferenceMin( const tnlVector< long int, tnlHost, int >& v1, const tnlVector< long int, tnlHost, int >& v2 );
#ifdef INSTANTIATE_FLOAT
template float       tnlVectorOperations< tnlHost >::getVectorDifferenceMin( const tnlVector< float, tnlHost, int >& v1,  const tnlVector< float, tnlHost, int >& v2);
#endif
template double      tnlVectorOperations< tnlHost >::getVectorDifferenceMin( const tnlVector< double, tnlHost, int >& v1, const tnlVector< double, tnlHost, int >& v2 );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double tnlVectorOperations< tnlHost >::getVectorDifferenceMin( const tnlVector< long double, tnlHost, int >& v1, const tnlVector< long double, tnlHost, int >& v2 );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         tnlVectorOperations< tnlHost >::getVectorDifferenceMin( const tnlVector< int, tnlHost, long int >& v1, const tnlVector< int, tnlHost, long int >& v2 );
template long int    tnlVectorOperations< tnlHost >::getVectorDifferenceMin( const tnlVector< long int, tnlHost, long int >& v1, const tnlVector< long int, tnlHost, long int >& v2 );
#ifdef INSTANTIATE_FLOAT
template float       tnlVectorOperations< tnlHost >::getVectorDifferenceMin( const tnlVector< float, tnlHost, long int >& v1, const tnlVector< float, tnlHost, long int >& v2 );
#endif
template double      tnlVectorOperations< tnlHost >::getVectorDifferenceMin( const tnlVector< double, tnlHost, long int >& v1, const tnlVector< double, tnlHost, long int >& v2 );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double tnlVectorOperations< tnlHost >::getVectorDifferenceMin( const tnlVector< long double, tnlHost, long int >& v1, const tnlVector< long double, tnlHost, long int >& v2 );
#endif
#endif

/****
 * Difference abs max
 */
template int         tnlVectorOperations< tnlHost >::getVectorDifferenceAbsMax( const tnlVector< int, tnlHost, int >& v1, const tnlVector< int, tnlHost, int >& v2 );
template long int    tnlVectorOperations< tnlHost >::getVectorDifferenceAbsMax( const tnlVector< long int, tnlHost, int >& v1, const tnlVector< long int, tnlHost, int >& v2 );
#ifdef INSTANTIATE_FLOAT
template float       tnlVectorOperations< tnlHost >::getVectorDifferenceAbsMax( const tnlVector< float, tnlHost, int >& v1,  const tnlVector< float, tnlHost, int >& v2);
#endif
template double      tnlVectorOperations< tnlHost >::getVectorDifferenceAbsMax( const tnlVector< double, tnlHost, int >& v1, const tnlVector< double, tnlHost, int >& v2 );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double tnlVectorOperations< tnlHost >::getVectorDifferenceAbsMax( const tnlVector< long double, tnlHost, int >& v1, const tnlVector< long double, tnlHost, int >& v2 );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         tnlVectorOperations< tnlHost >::getVectorDifferenceAbsMax( const tnlVector< int, tnlHost, long int >& v1, const tnlVector< int, tnlHost, long int >& v2 );
template long int    tnlVectorOperations< tnlHost >::getVectorDifferenceAbsMax( const tnlVector< long int, tnlHost, long int >& v1, const tnlVector< long int, tnlHost, long int >& v2 );
#ifdef INSTANTIATE_FLOAT
template float       tnlVectorOperations< tnlHost >::getVectorDifferenceAbsMax( const tnlVector< float, tnlHost, long int >& v1, const tnlVector< float, tnlHost, long int >& v2 );
#endif
template double      tnlVectorOperations< tnlHost >::getVectorDifferenceAbsMax( const tnlVector< double, tnlHost, long int >& v1, const tnlVector< double, tnlHost, long int >& v2 );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double tnlVectorOperations< tnlHost >::getVectorDifferenceAbsMax( const tnlVector< long double, tnlHost, long int >& v1, const tnlVector< long double, tnlHost, long int >& v2 );
#endif
#endif

/****
 * Difference abs min
 */
template int         tnlVectorOperations< tnlHost >::getVectorDifferenceAbsMin( const tnlVector< int, tnlHost, int >& v1, const tnlVector< int, tnlHost, int >& v2 );
template long int    tnlVectorOperations< tnlHost >::getVectorDifferenceAbsMin( const tnlVector< long int, tnlHost, int >& v1, const tnlVector< long int, tnlHost, int >& v2 );
#ifdef INSTANTIATE_FLOAT
template float       tnlVectorOperations< tnlHost >::getVectorDifferenceAbsMin( const tnlVector< float, tnlHost, int >& v1,  const tnlVector< float, tnlHost, int >& v2);
#endif
template double      tnlVectorOperations< tnlHost >::getVectorDifferenceAbsMin( const tnlVector< double, tnlHost, int >& v1, const tnlVector< double, tnlHost, int >& v2 );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double tnlVectorOperations< tnlHost >::getVectorDifferenceAbsMin( const tnlVector< long double, tnlHost, int >& v1, const tnlVector< long double, tnlHost, int >& v2 );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         tnlVectorOperations< tnlHost >::getVectorDifferenceAbsMin( const tnlVector< int, tnlHost, long int >& v1, const tnlVector< int, tnlHost, long int >& v2 );
template long int    tnlVectorOperations< tnlHost >::getVectorDifferenceAbsMin( const tnlVector< long int, tnlHost, long int >& v1, const tnlVector< long int, tnlHost, long int >& v2 );
#ifdef INSTANTIATE_FLOAT
template float       tnlVectorOperations< tnlHost >::getVectorDifferenceAbsMin( const tnlVector< float, tnlHost, long int >& v1, const tnlVector< float, tnlHost, long int >& v2 );
#endif
template double      tnlVectorOperations< tnlHost >::getVectorDifferenceAbsMin( const tnlVector< double, tnlHost, long int >& v1, const tnlVector< double, tnlHost, long int >& v2 );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double tnlVectorOperations< tnlHost >::getVectorDifferenceAbsMin( const tnlVector< long double, tnlHost, long int >& v1, const tnlVector< long double, tnlHost, long int >& v2 );
#endif
#endif


#endif

} // namespace Vectors
} // namespace TNL
