/***************************************************************************
                          VectorOperationsHost_impl.cpp  -  description
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
template int         VectorOperations< tnlHost >::getVectorMax( const Vector< int, tnlHost, int >& v );
template long int    VectorOperations< tnlHost >::getVectorMax( const Vector< long int, tnlHost, int >& v );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< tnlHost >::getVectorMax( const Vector< float, tnlHost, int >& v );
#endif
template double      VectorOperations< tnlHost >::getVectorMax( const Vector< double, tnlHost, int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< tnlHost >::getVectorMax( const Vector< long double, tnlHost, int >& v );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         VectorOperations< tnlHost >::getVectorMax( const Vector< int, tnlHost, long int >& v );
template long int    VectorOperations< tnlHost >::getVectorMax( const Vector< long int, tnlHost, long int >& v );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< tnlHost >::getVectorMax( const Vector< float, tnlHost, long int >& v );
#endif
template double      VectorOperations< tnlHost >::getVectorMax( const Vector< double, tnlHost, long int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< tnlHost >::getVectorMax( const Vector< long double, tnlHost, long int >& v );
#endif
#endif

/****
 * Min
 */
template int         VectorOperations< tnlHost >::getVectorMin( const Vector< int, tnlHost, int >& v );
template long int    VectorOperations< tnlHost >::getVectorMin( const Vector< long int, tnlHost, int >& v );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< tnlHost >::getVectorMin( const Vector< float, tnlHost, int >& v );
#endif
template double      VectorOperations< tnlHost >::getVectorMin( const Vector< double, tnlHost, int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< tnlHost >::getVectorMin( const Vector< long double, tnlHost, int >& v );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         VectorOperations< tnlHost >::getVectorMin( const Vector< int, tnlHost, long int >& v );
template long int    VectorOperations< tnlHost >::getVectorMin( const Vector< long int, tnlHost, long int >& v );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< tnlHost >::getVectorMin( const Vector< float, tnlHost, long int >& v );
#endif
template double      VectorOperations< tnlHost >::getVectorMin( const Vector< double, tnlHost, long int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< tnlHost >::getVectorMin( const Vector< long double, tnlHost, long int >& v );
#endif
#endif

/****
 * Abs max
 */
template int         VectorOperations< tnlHost >::getVectorAbsMax( const Vector< int, tnlHost, int >& v );
template long int    VectorOperations< tnlHost >::getVectorAbsMax( const Vector< long int, tnlHost, int >& v );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< tnlHost >::getVectorAbsMax( const Vector< float, tnlHost, int >& v );
#endif
template double      VectorOperations< tnlHost >::getVectorAbsMax( const Vector< double, tnlHost, int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< tnlHost >::getVectorAbsMax( const Vector< long double, tnlHost, int >& v );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         VectorOperations< tnlHost >::getVectorAbsMax( const Vector< int, tnlHost, long int >& v );
template long int    VectorOperations< tnlHost >::getVectorAbsMax( const Vector< long int, tnlHost, long int >& v );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< tnlHost >::getVectorAbsMax( const Vector< float, tnlHost, long int >& v );
#endif
template double      VectorOperations< tnlHost >::getVectorAbsMax( const Vector< double, tnlHost, long int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< tnlHost >::getVectorAbsMax( const Vector< long double, tnlHost, long int >& v );
#endif
#endif

/****
 * Abs min
 */
template int         VectorOperations< tnlHost >::getVectorAbsMin( const Vector< int, tnlHost, int >& v );
template long int    VectorOperations< tnlHost >::getVectorAbsMin( const Vector< long int, tnlHost, int >& v );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< tnlHost >::getVectorAbsMin( const Vector< float, tnlHost, int >& v );
#endif
template double      VectorOperations< tnlHost >::getVectorAbsMin( const Vector< double, tnlHost, int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< tnlHost >::getVectorAbsMin( const Vector< long double, tnlHost, int >& v );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         VectorOperations< tnlHost >::getVectorAbsMin( const Vector< int, tnlHost, long int >& v );
template long int    VectorOperations< tnlHost >::getVectorAbsMin( const Vector< long int, tnlHost, long int >& v );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< tnlHost >::getVectorAbsMin( const Vector< float, tnlHost, long int >& v );
#endif
template double      VectorOperations< tnlHost >::getVectorAbsMin( const Vector< double, tnlHost, long int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< tnlHost >::getVectorAbsMin( const Vector< long double, tnlHost, long int >& v );
#endif
#endif

/****
 * L1 norm
 */
template int         VectorOperations< tnlHost >::getVectorL1Norm( const Vector< int, tnlHost, int >& v );
template long int    VectorOperations< tnlHost >::getVectorL1Norm( const Vector< long int, tnlHost, int >& v );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< tnlHost >::getVectorL1Norm( const Vector< float, tnlHost, int >& v );
#endif
template double      VectorOperations< tnlHost >::getVectorL1Norm( const Vector< double, tnlHost, int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< tnlHost >::getVectorL1Norm( const Vector< long double, tnlHost, int >& v );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         VectorOperations< tnlHost >::getVectorL1Norm( const Vector< int, tnlHost, long int >& v );
template long int    VectorOperations< tnlHost >::getVectorL1Norm( const Vector< long int, tnlHost, long int >& v );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< tnlHost >::getVectorL1Norm( const Vector< float, tnlHost, long int >& v );
#endif
template double      VectorOperations< tnlHost >::getVectorL1Norm( const Vector< double, tnlHost, long int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< tnlHost >::getVectorL1Norm( const Vector< long double, tnlHost, long int >& v );
#endif
#endif

/****
 * L2 norm
 */
template int         VectorOperations< tnlHost >::getVectorL2Norm( const Vector< int, tnlHost, int >& v );
template long int    VectorOperations< tnlHost >::getVectorL2Norm( const Vector< long int, tnlHost, int >& v );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< tnlHost >::getVectorL2Norm( const Vector< float, tnlHost, int >& v );
#endif
template double      VectorOperations< tnlHost >::getVectorL2Norm( const Vector< double, tnlHost, int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< tnlHost >::getVectorL2Norm( const Vector< long double, tnlHost, int >& v );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         VectorOperations< tnlHost >::getVectorL2Norm( const Vector< int, tnlHost, long int >& v );
template long int    VectorOperations< tnlHost >::getVectorL2Norm( const Vector< long int, tnlHost, long int >& v );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< tnlHost >::getVectorL2Norm( const Vector< float, tnlHost, long int >& v );
#endif
template double      VectorOperations< tnlHost >::getVectorL2Norm( const Vector< double, tnlHost, long int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< tnlHost >::getVectorL2Norm( const Vector< long double, tnlHost, long int >& v );
#endif
#endif


/****
 * Lp norm
 */
template int         VectorOperations< tnlHost >::getVectorLpNorm( const Vector< int, tnlHost, int >& v, const int& p );
template long int    VectorOperations< tnlHost >::getVectorLpNorm( const Vector< long int, tnlHost, int >& v, const long int& p );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< tnlHost >::getVectorLpNorm( const Vector< float, tnlHost, int >& v, const float& p );
#endif
template double      VectorOperations< tnlHost >::getVectorLpNorm( const Vector< double, tnlHost, int >& v, const double& p );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< tnlHost >::getVectorLpNorm( const Vector< long double, tnlHost, int >& v, const long double& p );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         VectorOperations< tnlHost >::getVectorLpNorm( const Vector< int, tnlHost, long int >& v, const int& p );
template long int    VectorOperations< tnlHost >::getVectorLpNorm( const Vector< long int, tnlHost, long int >& v, const long int& p );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< tnlHost >::getVectorLpNorm( const Vector< float, tnlHost, long int >& v, const float& p );
#endif
template double      VectorOperations< tnlHost >::getVectorLpNorm( const Vector< double, tnlHost, long int >& v, const double& p );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< tnlHost >::getVectorLpNorm( const Vector< long double, tnlHost, long int >& v, const long double& p );
#endif
#endif



/****
 * Sum
 */
template int         VectorOperations< tnlHost >::getVectorSum( const Vector< int, tnlHost, int >& v );
template long int    VectorOperations< tnlHost >::getVectorSum( const Vector< long int, tnlHost, int >& v );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< tnlHost >::getVectorSum( const Vector< float, tnlHost, int >& v );
#endif
template double      VectorOperations< tnlHost >::getVectorSum( const Vector< double, tnlHost, int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< tnlHost >::getVectorSum( const Vector< long double, tnlHost, int >& v );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         VectorOperations< tnlHost >::getVectorSum( const Vector< int, tnlHost, long int >& v );
template long int    VectorOperations< tnlHost >::getVectorSum( const Vector< long int, tnlHost, long int >& v );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< tnlHost >::getVectorSum( const Vector< float, tnlHost, long int >& v );
#endif
template double      VectorOperations< tnlHost >::getVectorSum( const Vector< double, tnlHost, long int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< tnlHost >::getVectorSum( const Vector< long double, tnlHost, long int >& v );
#endif
#endif

/****
 * Difference max
 */
template int         VectorOperations< tnlHost >::getVectorDifferenceMax( const Vector< int, tnlHost, int >& v1, const Vector< int, tnlHost, int >& v2 );
template long int    VectorOperations< tnlHost >::getVectorDifferenceMax( const Vector< long int, tnlHost, int >& v1, const Vector< long int, tnlHost, int >& v2 );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< tnlHost >::getVectorDifferenceMax( const Vector< float, tnlHost, int >& v1,  const Vector< float, tnlHost, int >& v2);
#endif
template double      VectorOperations< tnlHost >::getVectorDifferenceMax( const Vector< double, tnlHost, int >& v1, const Vector< double, tnlHost, int >& v2 );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< tnlHost >::getVectorDifferenceMax( const Vector< long double, tnlHost, int >& v1, const Vector< long double, tnlHost, int >& v2 );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         VectorOperations< tnlHost >::getVectorDifferenceMax( const Vector< int, tnlHost, long int >& v1, const Vector< int, tnlHost, long int >& v2 );
template long int    VectorOperations< tnlHost >::getVectorDifferenceMax( const Vector< long int, tnlHost, long int >& v1, const Vector< long int, tnlHost, long int >& v2 );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< tnlHost >::getVectorDifferenceMax( const Vector< float, tnlHost, long int >& v1, const Vector< float, tnlHost, long int >& v2 );
#endif
template double      VectorOperations< tnlHost >::getVectorDifferenceMax( const Vector< double, tnlHost, long int >& v1, const Vector< double, tnlHost, long int >& v2 );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< tnlHost >::getVectorDifferenceMax( const Vector< long double, tnlHost, long int >& v1, const Vector< long double, tnlHost, long int >& v2 );
#endif
#endif

/****
 * Difference min
 */
template int         VectorOperations< tnlHost >::getVectorDifferenceMin( const Vector< int, tnlHost, int >& v1, const Vector< int, tnlHost, int >& v2 );
template long int    VectorOperations< tnlHost >::getVectorDifferenceMin( const Vector< long int, tnlHost, int >& v1, const Vector< long int, tnlHost, int >& v2 );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< tnlHost >::getVectorDifferenceMin( const Vector< float, tnlHost, int >& v1,  const Vector< float, tnlHost, int >& v2);
#endif
template double      VectorOperations< tnlHost >::getVectorDifferenceMin( const Vector< double, tnlHost, int >& v1, const Vector< double, tnlHost, int >& v2 );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< tnlHost >::getVectorDifferenceMin( const Vector< long double, tnlHost, int >& v1, const Vector< long double, tnlHost, int >& v2 );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         VectorOperations< tnlHost >::getVectorDifferenceMin( const Vector< int, tnlHost, long int >& v1, const Vector< int, tnlHost, long int >& v2 );
template long int    VectorOperations< tnlHost >::getVectorDifferenceMin( const Vector< long int, tnlHost, long int >& v1, const Vector< long int, tnlHost, long int >& v2 );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< tnlHost >::getVectorDifferenceMin( const Vector< float, tnlHost, long int >& v1, const Vector< float, tnlHost, long int >& v2 );
#endif
template double      VectorOperations< tnlHost >::getVectorDifferenceMin( const Vector< double, tnlHost, long int >& v1, const Vector< double, tnlHost, long int >& v2 );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< tnlHost >::getVectorDifferenceMin( const Vector< long double, tnlHost, long int >& v1, const Vector< long double, tnlHost, long int >& v2 );
#endif
#endif

/****
 * Difference abs max
 */
template int         VectorOperations< tnlHost >::getVectorDifferenceAbsMax( const Vector< int, tnlHost, int >& v1, const Vector< int, tnlHost, int >& v2 );
template long int    VectorOperations< tnlHost >::getVectorDifferenceAbsMax( const Vector< long int, tnlHost, int >& v1, const Vector< long int, tnlHost, int >& v2 );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< tnlHost >::getVectorDifferenceAbsMax( const Vector< float, tnlHost, int >& v1,  const Vector< float, tnlHost, int >& v2);
#endif
template double      VectorOperations< tnlHost >::getVectorDifferenceAbsMax( const Vector< double, tnlHost, int >& v1, const Vector< double, tnlHost, int >& v2 );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< tnlHost >::getVectorDifferenceAbsMax( const Vector< long double, tnlHost, int >& v1, const Vector< long double, tnlHost, int >& v2 );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         VectorOperations< tnlHost >::getVectorDifferenceAbsMax( const Vector< int, tnlHost, long int >& v1, const Vector< int, tnlHost, long int >& v2 );
template long int    VectorOperations< tnlHost >::getVectorDifferenceAbsMax( const Vector< long int, tnlHost, long int >& v1, const Vector< long int, tnlHost, long int >& v2 );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< tnlHost >::getVectorDifferenceAbsMax( const Vector< float, tnlHost, long int >& v1, const Vector< float, tnlHost, long int >& v2 );
#endif
template double      VectorOperations< tnlHost >::getVectorDifferenceAbsMax( const Vector< double, tnlHost, long int >& v1, const Vector< double, tnlHost, long int >& v2 );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< tnlHost >::getVectorDifferenceAbsMax( const Vector< long double, tnlHost, long int >& v1, const Vector< long double, tnlHost, long int >& v2 );
#endif
#endif

/****
 * Difference abs min
 */
template int         VectorOperations< tnlHost >::getVectorDifferenceAbsMin( const Vector< int, tnlHost, int >& v1, const Vector< int, tnlHost, int >& v2 );
template long int    VectorOperations< tnlHost >::getVectorDifferenceAbsMin( const Vector< long int, tnlHost, int >& v1, const Vector< long int, tnlHost, int >& v2 );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< tnlHost >::getVectorDifferenceAbsMin( const Vector< float, tnlHost, int >& v1,  const Vector< float, tnlHost, int >& v2);
#endif
template double      VectorOperations< tnlHost >::getVectorDifferenceAbsMin( const Vector< double, tnlHost, int >& v1, const Vector< double, tnlHost, int >& v2 );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< tnlHost >::getVectorDifferenceAbsMin( const Vector< long double, tnlHost, int >& v1, const Vector< long double, tnlHost, int >& v2 );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         VectorOperations< tnlHost >::getVectorDifferenceAbsMin( const Vector< int, tnlHost, long int >& v1, const Vector< int, tnlHost, long int >& v2 );
template long int    VectorOperations< tnlHost >::getVectorDifferenceAbsMin( const Vector< long int, tnlHost, long int >& v1, const Vector< long int, tnlHost, long int >& v2 );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< tnlHost >::getVectorDifferenceAbsMin( const Vector< float, tnlHost, long int >& v1, const Vector< float, tnlHost, long int >& v2 );
#endif
template double      VectorOperations< tnlHost >::getVectorDifferenceAbsMin( const Vector< double, tnlHost, long int >& v1, const Vector< double, tnlHost, long int >& v2 );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< tnlHost >::getVectorDifferenceAbsMin( const Vector< long double, tnlHost, long int >& v1, const Vector< long double, tnlHost, long int >& v2 );
#endif
#endif


#endif

} // namespace Vectors
} // namespace TNL
