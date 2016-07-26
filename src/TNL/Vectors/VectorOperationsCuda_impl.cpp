/***************************************************************************
                          VectorOperationsCuda_impl.cpp  -  description
                             -------------------
    begin                : Dec 10, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
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
template int         VectorOperations< tnlCuda >::getVectorMax( const Vector< int, tnlCuda, int >& v );
template long int    VectorOperations< tnlCuda >::getVectorMax( const Vector< long int, tnlCuda, int >& v );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< tnlCuda >::getVectorMax( const Vector< float, tnlCuda, int >& v );
#endif
template double      VectorOperations< tnlCuda >::getVectorMax( const Vector< double, tnlCuda, int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< tnlCuda >::getVectorMax( const Vector< long double, tnlCuda, int >& v );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         VectorOperations< tnlCuda >::getVectorMax( const Vector< int, tnlCuda, long int >& v );
template long int    VectorOperations< tnlCuda >::getVectorMax( const Vector< long int, tnlCuda, long int >& v );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< tnlCuda >::getVectorMax( const Vector< float, tnlCuda, long int >& v );
#endif
template double      VectorOperations< tnlCuda >::getVectorMax( const Vector< double, tnlCuda, long int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< tnlCuda >::getVectorMax( const Vector< long double, tnlCuda, long int >& v );
#endif
#endif

/****
 * Min
 */
template int         VectorOperations< tnlCuda >::getVectorMin( const Vector< int, tnlCuda, int >& v );
template long int    VectorOperations< tnlCuda >::getVectorMin( const Vector< long int, tnlCuda, int >& v );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< tnlCuda >::getVectorMin( const Vector< float, tnlCuda, int >& v );
#endif
template double      VectorOperations< tnlCuda >::getVectorMin( const Vector< double, tnlCuda, int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< tnlCuda >::getVectorMin( const Vector< long double, tnlCuda, int >& v );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         VectorOperations< tnlCuda >::getVectorMin( const Vector< int, tnlCuda, long int >& v );
template long int    VectorOperations< tnlCuda >::getVectorMin( const Vector< long int, tnlCuda, long int >& v );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< tnlCuda >::getVectorMin( const Vector< float, tnlCuda, long int >& v );
#endif
template double      VectorOperations< tnlCuda >::getVectorMin( const Vector< double, tnlCuda, long int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< tnlCuda >::getVectorMin( const Vector< long double, tnlCuda, long int >& v );
#endif
#endif

/****
 * Abs max
 */
template int         VectorOperations< tnlCuda >::getVectorAbsMax( const Vector< int, tnlCuda, int >& v );
template long int    VectorOperations< tnlCuda >::getVectorAbsMax( const Vector< long int, tnlCuda, int >& v );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< tnlCuda >::getVectorAbsMax( const Vector< float, tnlCuda, int >& v );
#endif
template double      VectorOperations< tnlCuda >::getVectorAbsMax( const Vector< double, tnlCuda, int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< tnlCuda >::getVectorAbsMax( const Vector< long double, tnlCuda, int >& v );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         VectorOperations< tnlCuda >::getVectorAbsMax( const Vector< int, tnlCuda, long int >& v );
template long int    VectorOperations< tnlCuda >::getVectorAbsMax( const Vector< long int, tnlCuda, long int >& v );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< tnlCuda >::getVectorAbsMax( const Vector< float, tnlCuda, long int >& v );
#endif
template double      VectorOperations< tnlCuda >::getVectorAbsMax( const Vector< double, tnlCuda, long int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< tnlCuda >::getVectorAbsMax( const Vector< long double, tnlCuda, long int >& v );
#endif
#endif


/****
 * Abs min
 */
template int         VectorOperations< tnlCuda >::getVectorAbsMin( const Vector< int, tnlCuda, int >& v );
template long int    VectorOperations< tnlCuda >::getVectorAbsMin( const Vector< long int, tnlCuda, int >& v );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< tnlCuda >::getVectorAbsMin( const Vector< float, tnlCuda, int >& v );
#endif
template double      VectorOperations< tnlCuda >::getVectorAbsMin( const Vector< double, tnlCuda, int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< tnlCuda >::getVectorAbsMin( const Vector< long double, tnlCuda, int >& v );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         VectorOperations< tnlCuda >::getVectorAbsMin( const Vector< int, tnlCuda, long int >& v );
template long int    VectorOperations< tnlCuda >::getVectorAbsMin( const Vector< long int, tnlCuda, long int >& v );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< tnlCuda >::getVectorAbsMin( const Vector< float, tnlCuda, long int >& v );
#endif
template double      VectorOperations< tnlCuda >::getVectorAbsMin( const Vector< double, tnlCuda, long int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< tnlCuda >::getVectorAbsMin( const Vector< long double, tnlCuda, long int >& v );
#endif
#endif

/****
 * L2 norm
 */
template int         VectorOperations< tnlCuda >::getVectorL2Norm( const Vector< int, tnlCuda, int >& v );
template long int    VectorOperations< tnlCuda >::getVectorL2Norm( const Vector< long int, tnlCuda, int >& v );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< tnlCuda >::getVectorL2Norm( const Vector< float, tnlCuda, int >& v );
#endif
template double      VectorOperations< tnlCuda >::getVectorL2Norm( const Vector< double, tnlCuda, int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< tnlCuda >::getVectorL2Norm( const Vector< long double, tnlCuda, int >& v );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         VectorOperations< tnlCuda >::getVectorL2Norm( const Vector< int, tnlCuda, long int >& v );
template long int    VectorOperations< tnlCuda >::getVectorL2Norm( const Vector< long int, tnlCuda, long int >& v );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< tnlCuda >::getVectorL2Norm( const Vector< float, tnlCuda, long int >& v );
#endif
template double      VectorOperations< tnlCuda >::getVectorL2Norm( const Vector< double, tnlCuda, long int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< tnlCuda >::getVectorL2Norm( const Vector< long double, tnlCuda, long int >& v );
#endif
#endif

/****
 * L1 norm
 */
template int         VectorOperations< tnlCuda >::getVectorL1Norm( const Vector< int, tnlCuda, int >& v );
template long int    VectorOperations< tnlCuda >::getVectorL1Norm( const Vector< long int, tnlCuda, int >& v );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< tnlCuda >::getVectorL1Norm( const Vector< float, tnlCuda, int >& v );
#endif
template double      VectorOperations< tnlCuda >::getVectorL1Norm( const Vector< double, tnlCuda, int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< tnlCuda >::getVectorL1Norm( const Vector< long double, tnlCuda, int >& v );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         VectorOperations< tnlCuda >::getVectorL1Norm( const Vector< int, tnlCuda, long int >& v );
template long int    VectorOperations< tnlCuda >::getVectorL1Norm( const Vector< long int, tnlCuda, long int >& v );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< tnlCuda >::getVectorL1Norm( const Vector< float, tnlCuda, long int >& v );
#endif
template double      VectorOperations< tnlCuda >::getVectorL1Norm( const Vector< double, tnlCuda, long int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< tnlCuda >::getVectorL1Norm( const Vector< long double, tnlCuda, long int >& v );
#endif
#endif

/****
 * Lp norm
 */
template int         VectorOperations< tnlCuda >::getVectorLpNorm( const Vector< int, tnlCuda, int >& v, const int& p );
template long int    VectorOperations< tnlCuda >::getVectorLpNorm( const Vector< long int, tnlCuda, int >& v, const long int& p );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< tnlCuda >::getVectorLpNorm( const Vector< float, tnlCuda, int >& v, const float& p );
#endif
template double      VectorOperations< tnlCuda >::getVectorLpNorm( const Vector< double, tnlCuda, int >& v, const double& p );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< tnlCuda >::getVectorLpNorm( const Vector< long double, tnlCuda, int >& v, const long double& p );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         VectorOperations< tnlCuda >::getVectorLpNorm( const Vector< int, tnlCuda, long int >& v, const int& p );
template long int    VectorOperations< tnlCuda >::getVectorLpNorm( const Vector< long int, tnlCuda, long int >& v, const long int& p );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< tnlCuda >::getVectorLpNorm( const Vector< float, tnlCuda, long int >& v, const float& p );
#endif
template double      VectorOperations< tnlCuda >::getVectorLpNorm( const Vector< double, tnlCuda, long int >& v, const double& p );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< tnlCuda >::getVectorLpNorm( const Vector< long double, tnlCuda, long int >& v, const long double& p );
#endif
#endif



/****
 * Sum
 */
template int         VectorOperations< tnlCuda >::getVectorSum( const Vector< int, tnlCuda, int >& v );
template long int    VectorOperations< tnlCuda >::getVectorSum( const Vector< long int, tnlCuda, int >& v );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< tnlCuda >::getVectorSum( const Vector< float, tnlCuda, int >& v );
#endif
template double      VectorOperations< tnlCuda >::getVectorSum( const Vector< double, tnlCuda, int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< tnlCuda >::getVectorSum( const Vector< long double, tnlCuda, int >& v );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         VectorOperations< tnlCuda >::getVectorSum( const Vector< int, tnlCuda, long int >& v );
template long int    VectorOperations< tnlCuda >::getVectorSum( const Vector< long int, tnlCuda, long int >& v );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< tnlCuda >::getVectorSum( const Vector< float, tnlCuda, long int >& v );
#endif
template double      VectorOperations< tnlCuda >::getVectorSum( const Vector< double, tnlCuda, long int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< tnlCuda >::getVectorSum( const Vector< long double, tnlCuda, long int >& v );
#endif
#endif

/****
 * Difference max
 */
template int         VectorOperations< tnlCuda >::getVectorDifferenceMax( const Vector< int, tnlCuda, int >& v1, const Vector< int, tnlCuda, int >& v2 );
template long int    VectorOperations< tnlCuda >::getVectorDifferenceMax( const Vector< long int, tnlCuda, int >& v1, const Vector< long int, tnlCuda, int >& v2 );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< tnlCuda >::getVectorDifferenceMax( const Vector< float, tnlCuda, int >& v1,  const Vector< float, tnlCuda, int >& v2);
#endif
template double      VectorOperations< tnlCuda >::getVectorDifferenceMax( const Vector< double, tnlCuda, int >& v1, const Vector< double, tnlCuda, int >& v2 );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< tnlCuda >::getVectorDifferenceMax( const Vector< long double, tnlCuda, int >& v1, const Vector< long double, tnlCuda, int >& v2 );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         VectorOperations< tnlCuda >::getVectorDifferenceMax( const Vector< int, tnlCuda, long int >& v1, const Vector< int, tnlCuda, long int >& v2 );
template long int    VectorOperations< tnlCuda >::getVectorDifferenceMax( const Vector< long int, tnlCuda, long int >& v1, const Vector< long int, tnlCuda, long int >& v2 );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< tnlCuda >::getVectorDifferenceMax( const Vector< float, tnlCuda, long int >& v1, const Vector< float, tnlCuda, long int >& v2 );
#endif
template double      VectorOperations< tnlCuda >::getVectorDifferenceMax( const Vector< double, tnlCuda, long int >& v1, const Vector< double, tnlCuda, long int >& v2 );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< tnlCuda >::getVectorDifferenceMax( const Vector< long double, tnlCuda, long int >& v1, const Vector< long double, tnlCuda, long int >& v2 );
#endif
#endif

/****
 * Difference min
 */
template int         VectorOperations< tnlCuda >::getVectorDifferenceMin( const Vector< int, tnlCuda, int >& v1, const Vector< int, tnlCuda, int >& v2 );
template long int    VectorOperations< tnlCuda >::getVectorDifferenceMin( const Vector< long int, tnlCuda, int >& v1, const Vector< long int, tnlCuda, int >& v2 );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< tnlCuda >::getVectorDifferenceMin( const Vector< float, tnlCuda, int >& v1,  const Vector< float, tnlCuda, int >& v2);
#endif
template double      VectorOperations< tnlCuda >::getVectorDifferenceMin( const Vector< double, tnlCuda, int >& v1, const Vector< double, tnlCuda, int >& v2 );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< tnlCuda >::getVectorDifferenceMin( const Vector< long double, tnlCuda, int >& v1, const Vector< long double, tnlCuda, int >& v2 );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         VectorOperations< tnlCuda >::getVectorDifferenceMin( const Vector< int, tnlCuda, long int >& v1, const Vector< int, tnlCuda, long int >& v2 );
template long int    VectorOperations< tnlCuda >::getVectorDifferenceMin( const Vector< long int, tnlCuda, long int >& v1, const Vector< long int, tnlCuda, long int >& v2 );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< tnlCuda >::getVectorDifferenceMin( const Vector< float, tnlCuda, long int >& v1, const Vector< float, tnlCuda, long int >& v2 );
#endif
template double      VectorOperations< tnlCuda >::getVectorDifferenceMin( const Vector< double, tnlCuda, long int >& v1, const Vector< double, tnlCuda, long int >& v2 );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< tnlCuda >::getVectorDifferenceMin( const Vector< long double, tnlCuda, long int >& v1, const Vector< long double, tnlCuda, long int >& v2 );
#endif
#endif

/****
 * Difference abs max
 */
template int         VectorOperations< tnlCuda >::getVectorDifferenceAbsMax( const Vector< int, tnlCuda, int >& v1, const Vector< int, tnlCuda, int >& v2 );
template long int    VectorOperations< tnlCuda >::getVectorDifferenceAbsMax( const Vector< long int, tnlCuda, int >& v1, const Vector< long int, tnlCuda, int >& v2 );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< tnlCuda >::getVectorDifferenceAbsMax( const Vector< float, tnlCuda, int >& v1,  const Vector< float, tnlCuda, int >& v2);
#endif
template double      VectorOperations< tnlCuda >::getVectorDifferenceAbsMax( const Vector< double, tnlCuda, int >& v1, const Vector< double, tnlCuda, int >& v2 );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< tnlCuda >::getVectorDifferenceAbsMax( const Vector< long double, tnlCuda, int >& v1, const Vector< long double, tnlCuda, int >& v2 );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         VectorOperations< tnlCuda >::getVectorDifferenceAbsMax( const Vector< int, tnlCuda, long int >& v1, const Vector< int, tnlCuda, long int >& v2 );
template long int    VectorOperations< tnlCuda >::getVectorDifferenceAbsMax( const Vector< long int, tnlCuda, long int >& v1, const Vector< long int, tnlCuda, long int >& v2 );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< tnlCuda >::getVectorDifferenceAbsMax( const Vector< float, tnlCuda, long int >& v1, const Vector< float, tnlCuda, long int >& v2 );
#endif
template double      VectorOperations< tnlCuda >::getVectorDifferenceAbsMax( const Vector< double, tnlCuda, long int >& v1, const Vector< double, tnlCuda, long int >& v2 );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< tnlCuda >::getVectorDifferenceAbsMax( const Vector< long double, tnlCuda, long int >& v1, const Vector< long double, tnlCuda, long int >& v2 );
#endif
#endif


/****
 * Difference abs min
 */
template int         VectorOperations< tnlCuda >::getVectorDifferenceAbsMin( const Vector< int, tnlCuda, int >& v1, const Vector< int, tnlCuda, int >& v2 );
template long int    VectorOperations< tnlCuda >::getVectorDifferenceAbsMin( const Vector< long int, tnlCuda, int >& v1, const Vector< long int, tnlCuda, int >& v2 );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< tnlCuda >::getVectorDifferenceAbsMin( const Vector< float, tnlCuda, int >& v1,  const Vector< float, tnlCuda, int >& v2);
#endif
template double      VectorOperations< tnlCuda >::getVectorDifferenceAbsMin( const Vector< double, tnlCuda, int >& v1, const Vector< double, tnlCuda, int >& v2 );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< tnlCuda >::getVectorDifferenceAbsMin( const Vector< long double, tnlCuda, int >& v1, const Vector< long double, tnlCuda, int >& v2 );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         VectorOperations< tnlCuda >::getVectorDifferenceAbsMin( const Vector< int, tnlCuda, long int >& v1, const Vector< int, tnlCuda, long int >& v2 );
template long int    VectorOperations< tnlCuda >::getVectorDifferenceAbsMin( const Vector< long int, tnlCuda, long int >& v1, const Vector< long int, tnlCuda, long int >& v2 );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< tnlCuda >::getVectorDifferenceAbsMin( const Vector< float, tnlCuda, long int >& v1, const Vector< float, tnlCuda, long int >& v2 );
#endif
template double      VectorOperations< tnlCuda >::getVectorDifferenceAbsMin( const Vector< double, tnlCuda, long int >& v1, const Vector< double, tnlCuda, long int >& v2 );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< tnlCuda >::getVectorDifferenceAbsMin( const Vector< long double, tnlCuda, long int >& v1, const Vector< long double, tnlCuda, long int >& v2 );
#endif
#endif
 
#endif
 
} // namespace Vectors
} // namespace TNL