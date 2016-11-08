/***************************************************************************
                          VectorOperationsCuda_impl.cu  -  description
                             -------------------
    begin                : Jul 20, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/Containers/VectorOperations.h>

namespace TNL {
namespace Vectors {

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

/****
 * Max
 */
template int         VectorOperations< Devices::Cuda >::getVectorMax( const Vector< int, Devices::Cuda, int >& v );
template long int    VectorOperations< Devices::Cuda >::getVectorMax( const Vector< long int, Devices::Cuda, int >& v );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< Devices::Cuda >::getVectorMax( const Vector< float, Devices::Cuda, int >& v );
#endif
template double      VectorOperations< Devices::Cuda >::getVectorMax( const Vector< double, Devices::Cuda, int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< Devices::Cuda >::getVectorMax( const Vector< long double, Devices::Cuda, int >& v );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         VectorOperations< Devices::Cuda >::getVectorMax( const Vector< int, Devices::Cuda, long int >& v );
template long int    VectorOperations< Devices::Cuda >::getVectorMax( const Vector< long int, Devices::Cuda, long int >& v );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< Devices::Cuda >::getVectorMax( const Vector< float, Devices::Cuda, long int >& v );
#endif
template double      VectorOperations< Devices::Cuda >::getVectorMax( const Vector< double, Devices::Cuda, long int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< Devices::Cuda >::getVectorMax( const Vector< long double, Devices::Cuda, long int >& v );
#endif
#endif

/****
 * Min
 */
template int         VectorOperations< Devices::Cuda >::getVectorMin( const Vector< int, Devices::Cuda, int >& v );
template long int    VectorOperations< Devices::Cuda >::getVectorMin( const Vector< long int, Devices::Cuda, int >& v );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< Devices::Cuda >::getVectorMin( const Vector< float, Devices::Cuda, int >& v );
#endif
template double      VectorOperations< Devices::Cuda >::getVectorMin( const Vector< double, Devices::Cuda, int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< Devices::Cuda >::getVectorMin( const Vector< long double, Devices::Cuda, int >& v );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         VectorOperations< Devices::Cuda >::getVectorMin( const Vector< int, Devices::Cuda, long int >& v );
template long int    VectorOperations< Devices::Cuda >::getVectorMin( const Vector< long int, Devices::Cuda, long int >& v );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< Devices::Cuda >::getVectorMin( const Vector< float, Devices::Cuda, long int >& v );
#endif
template double      VectorOperations< Devices::Cuda >::getVectorMin( const Vector< double, Devices::Cuda, long int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< Devices::Cuda >::getVectorMin( const Vector< long double, Devices::Cuda, long int >& v );
#endif
#endif

/****
 * Abs max
 */
template int         VectorOperations< Devices::Cuda >::getVectorAbsMax( const Vector< int, Devices::Cuda, int >& v );
template long int    VectorOperations< Devices::Cuda >::getVectorAbsMax( const Vector< long int, Devices::Cuda, int >& v );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< Devices::Cuda >::getVectorAbsMax( const Vector< float, Devices::Cuda, int >& v );
#endif
template double      VectorOperations< Devices::Cuda >::getVectorAbsMax( const Vector< double, Devices::Cuda, int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< Devices::Cuda >::getVectorAbsMax( const Vector< long double, Devices::Cuda, int >& v );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         VectorOperations< Devices::Cuda >::getVectorAbsMax( const Vector< int, Devices::Cuda, long int >& v );
template long int    VectorOperations< Devices::Cuda >::getVectorAbsMax( const Vector< long int, Devices::Cuda, long int >& v );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< Devices::Cuda >::getVectorAbsMax( const Vector< float, Devices::Cuda, long int >& v );
#endif
template double      VectorOperations< Devices::Cuda >::getVectorAbsMax( const Vector< double, Devices::Cuda, long int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< Devices::Cuda >::getVectorAbsMax( const Vector< long double, Devices::Cuda, long int >& v );
#endif
#endif


/****
 * Abs min
 */
template int         VectorOperations< Devices::Cuda >::getVectorAbsMin( const Vector< int, Devices::Cuda, int >& v );
template long int    VectorOperations< Devices::Cuda >::getVectorAbsMin( const Vector< long int, Devices::Cuda, int >& v );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< Devices::Cuda >::getVectorAbsMin( const Vector< float, Devices::Cuda, int >& v );
#endif
template double      VectorOperations< Devices::Cuda >::getVectorAbsMin( const Vector< double, Devices::Cuda, int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< Devices::Cuda >::getVectorAbsMin( const Vector< long double, Devices::Cuda, int >& v );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         VectorOperations< Devices::Cuda >::getVectorAbsMin( const Vector< int, Devices::Cuda, long int >& v );
template long int    VectorOperations< Devices::Cuda >::getVectorAbsMin( const Vector< long int, Devices::Cuda, long int >& v );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< Devices::Cuda >::getVectorAbsMin( const Vector< float, Devices::Cuda, long int >& v );
#endif
template double      VectorOperations< Devices::Cuda >::getVectorAbsMin( const Vector< double, Devices::Cuda, long int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< Devices::Cuda >::getVectorAbsMin( const Vector< long double, Devices::Cuda, long int >& v );
#endif
#endif

/****
 * L2 norm
 */
template int         VectorOperations< Devices::Cuda >::getVectorL2Norm( const Vector< int, Devices::Cuda, int >& v );
template long int    VectorOperations< Devices::Cuda >::getVectorL2Norm( const Vector< long int, Devices::Cuda, int >& v );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< Devices::Cuda >::getVectorL2Norm( const Vector< float, Devices::Cuda, int >& v );
#endif
template double      VectorOperations< Devices::Cuda >::getVectorL2Norm( const Vector< double, Devices::Cuda, int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< Devices::Cuda >::getVectorL2Norm( const Vector< long double, Devices::Cuda, int >& v );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         VectorOperations< Devices::Cuda >::getVectorL2Norm( const Vector< int, Devices::Cuda, long int >& v );
template long int    VectorOperations< Devices::Cuda >::getVectorL2Norm( const Vector< long int, Devices::Cuda, long int >& v );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< Devices::Cuda >::getVectorL2Norm( const Vector< float, Devices::Cuda, long int >& v );
#endif
template double      VectorOperations< Devices::Cuda >::getVectorL2Norm( const Vector< double, Devices::Cuda, long int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< Devices::Cuda >::getVectorL2Norm( const Vector< long double, Devices::Cuda, long int >& v );
#endif
#endif

/****
 * L1 norm
 */
template int         VectorOperations< Devices::Cuda >::getVectorL1Norm( const Vector< int, Devices::Cuda, int >& v );
template long int    VectorOperations< Devices::Cuda >::getVectorL1Norm( const Vector< long int, Devices::Cuda, int >& v );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< Devices::Cuda >::getVectorL1Norm( const Vector< float, Devices::Cuda, int >& v );
#endif
template double      VectorOperations< Devices::Cuda >::getVectorL1Norm( const Vector< double, Devices::Cuda, int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< Devices::Cuda >::getVectorL1Norm( const Vector< long double, Devices::Cuda, int >& v );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         VectorOperations< Devices::Cuda >::getVectorL1Norm( const Vector< int, Devices::Cuda, long int >& v );
template long int    VectorOperations< Devices::Cuda >::getVectorL1Norm( const Vector< long int, Devices::Cuda, long int >& v );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< Devices::Cuda >::getVectorL1Norm( const Vector< float, Devices::Cuda, long int >& v );
#endif
template double      VectorOperations< Devices::Cuda >::getVectorL1Norm( const Vector< double, Devices::Cuda, long int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< Devices::Cuda >::getVectorL1Norm( const Vector< long double, Devices::Cuda, long int >& v );
#endif
#endif

/****
 * Lp norm
 */
template int         VectorOperations< Devices::Cuda >::getVectorLpNorm( const Vector< int, Devices::Cuda, int >& v, const int& p );
template long int    VectorOperations< Devices::Cuda >::getVectorLpNorm( const Vector< long int, Devices::Cuda, int >& v, const long int& p );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< Devices::Cuda >::getVectorLpNorm( const Vector< float, Devices::Cuda, int >& v, const float& p );
#endif
template double      VectorOperations< Devices::Cuda >::getVectorLpNorm( const Vector< double, Devices::Cuda, int >& v, const double& p );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< Devices::Cuda >::getVectorLpNorm( const Vector< long double, Devices::Cuda, int >& v, const long double& p );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         VectorOperations< Devices::Cuda >::getVectorLpNorm( const Vector< int, Devices::Cuda, long int >& v, const int& p );
template long int    VectorOperations< Devices::Cuda >::getVectorLpNorm( const Vector< long int, Devices::Cuda, long int >& v, const long int& p );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< Devices::Cuda >::getVectorLpNorm( const Vector< float, Devices::Cuda, long int >& v, const float& p );
#endif
template double      VectorOperations< Devices::Cuda >::getVectorLpNorm( const Vector< double, Devices::Cuda, long int >& v, const double& p );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< Devices::Cuda >::getVectorLpNorm( const Vector< long double, Devices::Cuda, long int >& v, const long double& p );
#endif
#endif



/****
 * Sum
 */
template int         VectorOperations< Devices::Cuda >::getVectorSum( const Vector< int, Devices::Cuda, int >& v );
template long int    VectorOperations< Devices::Cuda >::getVectorSum( const Vector< long int, Devices::Cuda, int >& v );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< Devices::Cuda >::getVectorSum( const Vector< float, Devices::Cuda, int >& v );
#endif
template double      VectorOperations< Devices::Cuda >::getVectorSum( const Vector< double, Devices::Cuda, int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< Devices::Cuda >::getVectorSum( const Vector< long double, Devices::Cuda, int >& v );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         VectorOperations< Devices::Cuda >::getVectorSum( const Vector< int, Devices::Cuda, long int >& v );
template long int    VectorOperations< Devices::Cuda >::getVectorSum( const Vector< long int, Devices::Cuda, long int >& v );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< Devices::Cuda >::getVectorSum( const Vector< float, Devices::Cuda, long int >& v );
#endif
template double      VectorOperations< Devices::Cuda >::getVectorSum( const Vector< double, Devices::Cuda, long int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< Devices::Cuda >::getVectorSum( const Vector< long double, Devices::Cuda, long int >& v );
#endif
#endif

/****
 * Difference max
 */
template int         VectorOperations< Devices::Cuda >::getVectorDifferenceMax( const Vector< int, Devices::Cuda, int >& v1, const Vector< int, Devices::Cuda, int >& v2 );
template long int    VectorOperations< Devices::Cuda >::getVectorDifferenceMax( const Vector< long int, Devices::Cuda, int >& v1, const Vector< long int, Devices::Cuda, int >& v2 );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< Devices::Cuda >::getVectorDifferenceMax( const Vector< float, Devices::Cuda, int >& v1,  const Vector< float, Devices::Cuda, int >& v2);
#endif
template double      VectorOperations< Devices::Cuda >::getVectorDifferenceMax( const Vector< double, Devices::Cuda, int >& v1, const Vector< double, Devices::Cuda, int >& v2 );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< Devices::Cuda >::getVectorDifferenceMax( const Vector< long double, Devices::Cuda, int >& v1, const Vector< long double, Devices::Cuda, int >& v2 );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         VectorOperations< Devices::Cuda >::getVectorDifferenceMax( const Vector< int, Devices::Cuda, long int >& v1, const Vector< int, Devices::Cuda, long int >& v2 );
template long int    VectorOperations< Devices::Cuda >::getVectorDifferenceMax( const Vector< long int, Devices::Cuda, long int >& v1, const Vector< long int, Devices::Cuda, long int >& v2 );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< Devices::Cuda >::getVectorDifferenceMax( const Vector< float, Devices::Cuda, long int >& v1, const Vector< float, Devices::Cuda, long int >& v2 );
#endif
template double      VectorOperations< Devices::Cuda >::getVectorDifferenceMax( const Vector< double, Devices::Cuda, long int >& v1, const Vector< double, Devices::Cuda, long int >& v2 );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< Devices::Cuda >::getVectorDifferenceMax( const Vector< long double, Devices::Cuda, long int >& v1, const Vector< long double, Devices::Cuda, long int >& v2 );
#endif
#endif

/****
 * Difference min
 */
template int         VectorOperations< Devices::Cuda >::getVectorDifferenceMin( const Vector< int, Devices::Cuda, int >& v1, const Vector< int, Devices::Cuda, int >& v2 );
template long int    VectorOperations< Devices::Cuda >::getVectorDifferenceMin( const Vector< long int, Devices::Cuda, int >& v1, const Vector< long int, Devices::Cuda, int >& v2 );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< Devices::Cuda >::getVectorDifferenceMin( const Vector< float, Devices::Cuda, int >& v1,  const Vector< float, Devices::Cuda, int >& v2);
#endif
template double      VectorOperations< Devices::Cuda >::getVectorDifferenceMin( const Vector< double, Devices::Cuda, int >& v1, const Vector< double, Devices::Cuda, int >& v2 );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< Devices::Cuda >::getVectorDifferenceMin( const Vector< long double, Devices::Cuda, int >& v1, const Vector< long double, Devices::Cuda, int >& v2 );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         VectorOperations< Devices::Cuda >::getVectorDifferenceMin( const Vector< int, Devices::Cuda, long int >& v1, const Vector< int, Devices::Cuda, long int >& v2 );
template long int    VectorOperations< Devices::Cuda >::getVectorDifferenceMin( const Vector< long int, Devices::Cuda, long int >& v1, const Vector< long int, Devices::Cuda, long int >& v2 );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< Devices::Cuda >::getVectorDifferenceMin( const Vector< float, Devices::Cuda, long int >& v1, const Vector< float, Devices::Cuda, long int >& v2 );
#endif
template double      VectorOperations< Devices::Cuda >::getVectorDifferenceMin( const Vector< double, Devices::Cuda, long int >& v1, const Vector< double, Devices::Cuda, long int >& v2 );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< Devices::Cuda >::getVectorDifferenceMin( const Vector< long double, Devices::Cuda, long int >& v1, const Vector< long double, Devices::Cuda, long int >& v2 );
#endif
#endif

/****
 * Difference abs max
 */
template int         VectorOperations< Devices::Cuda >::getVectorDifferenceAbsMax( const Vector< int, Devices::Cuda, int >& v1, const Vector< int, Devices::Cuda, int >& v2 );
template long int    VectorOperations< Devices::Cuda >::getVectorDifferenceAbsMax( const Vector< long int, Devices::Cuda, int >& v1, const Vector< long int, Devices::Cuda, int >& v2 );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< Devices::Cuda >::getVectorDifferenceAbsMax( const Vector< float, Devices::Cuda, int >& v1,  const Vector< float, Devices::Cuda, int >& v2);
#endif
template double      VectorOperations< Devices::Cuda >::getVectorDifferenceAbsMax( const Vector< double, Devices::Cuda, int >& v1, const Vector< double, Devices::Cuda, int >& v2 );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< Devices::Cuda >::getVectorDifferenceAbsMax( const Vector< long double, Devices::Cuda, int >& v1, const Vector< long double, Devices::Cuda, int >& v2 );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         VectorOperations< Devices::Cuda >::getVectorDifferenceAbsMax( const Vector< int, Devices::Cuda, long int >& v1, const Vector< int, Devices::Cuda, long int >& v2 );
template long int    VectorOperations< Devices::Cuda >::getVectorDifferenceAbsMax( const Vector< long int, Devices::Cuda, long int >& v1, const Vector< long int, Devices::Cuda, long int >& v2 );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< Devices::Cuda >::getVectorDifferenceAbsMax( const Vector< float, Devices::Cuda, long int >& v1, const Vector< float, Devices::Cuda, long int >& v2 );
#endif
template double      VectorOperations< Devices::Cuda >::getVectorDifferenceAbsMax( const Vector< double, Devices::Cuda, long int >& v1, const Vector< double, Devices::Cuda, long int >& v2 );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< Devices::Cuda >::getVectorDifferenceAbsMax( const Vector< long double, Devices::Cuda, long int >& v1, const Vector< long double, Devices::Cuda, long int >& v2 );
#endif
#endif


/****
 * Difference abs min
 */
template int         VectorOperations< Devices::Cuda >::getVectorDifferenceAbsMin( const Vector< int, Devices::Cuda, int >& v1, const Vector< int, Devices::Cuda, int >& v2 );
template long int    VectorOperations< Devices::Cuda >::getVectorDifferenceAbsMin( const Vector< long int, Devices::Cuda, int >& v1, const Vector< long int, Devices::Cuda, int >& v2 );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< Devices::Cuda >::getVectorDifferenceAbsMin( const Vector< float, Devices::Cuda, int >& v1,  const Vector< float, Devices::Cuda, int >& v2);
#endif
template double      VectorOperations< Devices::Cuda >::getVectorDifferenceAbsMin( const Vector< double, Devices::Cuda, int >& v1, const Vector< double, Devices::Cuda, int >& v2 );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< Devices::Cuda >::getVectorDifferenceAbsMin( const Vector< long double, Devices::Cuda, int >& v1, const Vector< long double, Devices::Cuda, int >& v2 );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         VectorOperations< Devices::Cuda >::getVectorDifferenceAbsMin( const Vector< int, Devices::Cuda, long int >& v1, const Vector< int, Devices::Cuda, long int >& v2 );
template long int    VectorOperations< Devices::Cuda >::getVectorDifferenceAbsMin( const Vector< long int, Devices::Cuda, long int >& v1, const Vector< long int, Devices::Cuda, long int >& v2 );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< Devices::Cuda >::getVectorDifferenceAbsMin( const Vector< float, Devices::Cuda, long int >& v1, const Vector< float, Devices::Cuda, long int >& v2 );
#endif
template double      VectorOperations< Devices::Cuda >::getVectorDifferenceAbsMin( const Vector< double, Devices::Cuda, long int >& v1, const Vector< double, Devices::Cuda, long int >& v2 );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< Devices::Cuda >::getVectorDifferenceAbsMin( const Vector< long double, Devices::Cuda, long int >& v1, const Vector< long double, Devices::Cuda, long int >& v2 );
#endif
#endif
 
#endif
 
} // namespace Vectors
} // namespace TNL
