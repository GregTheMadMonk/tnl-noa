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
template int         VectorOperations< Devices::Host >::getVectorMax( const Vector< int, Devices::Host, int >& v );
template long int    VectorOperations< Devices::Host >::getVectorMax( const Vector< long int, Devices::Host, int >& v );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< Devices::Host >::getVectorMax( const Vector< float, Devices::Host, int >& v );
#endif
template double      VectorOperations< Devices::Host >::getVectorMax( const Vector< double, Devices::Host, int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< Devices::Host >::getVectorMax( const Vector< long double, Devices::Host, int >& v );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         VectorOperations< Devices::Host >::getVectorMax( const Vector< int, Devices::Host, long int >& v );
template long int    VectorOperations< Devices::Host >::getVectorMax( const Vector< long int, Devices::Host, long int >& v );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< Devices::Host >::getVectorMax( const Vector< float, Devices::Host, long int >& v );
#endif
template double      VectorOperations< Devices::Host >::getVectorMax( const Vector< double, Devices::Host, long int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< Devices::Host >::getVectorMax( const Vector< long double, Devices::Host, long int >& v );
#endif
#endif

/****
 * Min
 */
template int         VectorOperations< Devices::Host >::getVectorMin( const Vector< int, Devices::Host, int >& v );
template long int    VectorOperations< Devices::Host >::getVectorMin( const Vector< long int, Devices::Host, int >& v );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< Devices::Host >::getVectorMin( const Vector< float, Devices::Host, int >& v );
#endif
template double      VectorOperations< Devices::Host >::getVectorMin( const Vector< double, Devices::Host, int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< Devices::Host >::getVectorMin( const Vector< long double, Devices::Host, int >& v );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         VectorOperations< Devices::Host >::getVectorMin( const Vector< int, Devices::Host, long int >& v );
template long int    VectorOperations< Devices::Host >::getVectorMin( const Vector< long int, Devices::Host, long int >& v );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< Devices::Host >::getVectorMin( const Vector< float, Devices::Host, long int >& v );
#endif
template double      VectorOperations< Devices::Host >::getVectorMin( const Vector< double, Devices::Host, long int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< Devices::Host >::getVectorMin( const Vector< long double, Devices::Host, long int >& v );
#endif
#endif

/****
 * Abs max
 */
template int         VectorOperations< Devices::Host >::getVectorAbsMax( const Vector< int, Devices::Host, int >& v );
template long int    VectorOperations< Devices::Host >::getVectorAbsMax( const Vector< long int, Devices::Host, int >& v );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< Devices::Host >::getVectorAbsMax( const Vector< float, Devices::Host, int >& v );
#endif
template double      VectorOperations< Devices::Host >::getVectorAbsMax( const Vector< double, Devices::Host, int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< Devices::Host >::getVectorAbsMax( const Vector< long double, Devices::Host, int >& v );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         VectorOperations< Devices::Host >::getVectorAbsMax( const Vector< int, Devices::Host, long int >& v );
template long int    VectorOperations< Devices::Host >::getVectorAbsMax( const Vector< long int, Devices::Host, long int >& v );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< Devices::Host >::getVectorAbsMax( const Vector< float, Devices::Host, long int >& v );
#endif
template double      VectorOperations< Devices::Host >::getVectorAbsMax( const Vector< double, Devices::Host, long int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< Devices::Host >::getVectorAbsMax( const Vector< long double, Devices::Host, long int >& v );
#endif
#endif

/****
 * Abs min
 */
template int         VectorOperations< Devices::Host >::getVectorAbsMin( const Vector< int, Devices::Host, int >& v );
template long int    VectorOperations< Devices::Host >::getVectorAbsMin( const Vector< long int, Devices::Host, int >& v );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< Devices::Host >::getVectorAbsMin( const Vector< float, Devices::Host, int >& v );
#endif
template double      VectorOperations< Devices::Host >::getVectorAbsMin( const Vector< double, Devices::Host, int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< Devices::Host >::getVectorAbsMin( const Vector< long double, Devices::Host, int >& v );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         VectorOperations< Devices::Host >::getVectorAbsMin( const Vector< int, Devices::Host, long int >& v );
template long int    VectorOperations< Devices::Host >::getVectorAbsMin( const Vector< long int, Devices::Host, long int >& v );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< Devices::Host >::getVectorAbsMin( const Vector< float, Devices::Host, long int >& v );
#endif
template double      VectorOperations< Devices::Host >::getVectorAbsMin( const Vector< double, Devices::Host, long int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< Devices::Host >::getVectorAbsMin( const Vector< long double, Devices::Host, long int >& v );
#endif
#endif

/****
 * L1 norm
 */
template int         VectorOperations< Devices::Host >::getVectorL1Norm( const Vector< int, Devices::Host, int >& v );
template long int    VectorOperations< Devices::Host >::getVectorL1Norm( const Vector< long int, Devices::Host, int >& v );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< Devices::Host >::getVectorL1Norm( const Vector< float, Devices::Host, int >& v );
#endif
template double      VectorOperations< Devices::Host >::getVectorL1Norm( const Vector< double, Devices::Host, int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< Devices::Host >::getVectorL1Norm( const Vector< long double, Devices::Host, int >& v );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         VectorOperations< Devices::Host >::getVectorL1Norm( const Vector< int, Devices::Host, long int >& v );
template long int    VectorOperations< Devices::Host >::getVectorL1Norm( const Vector< long int, Devices::Host, long int >& v );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< Devices::Host >::getVectorL1Norm( const Vector< float, Devices::Host, long int >& v );
#endif
template double      VectorOperations< Devices::Host >::getVectorL1Norm( const Vector< double, Devices::Host, long int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< Devices::Host >::getVectorL1Norm( const Vector< long double, Devices::Host, long int >& v );
#endif
#endif

/****
 * L2 norm
 */
template int         VectorOperations< Devices::Host >::getVectorL2Norm( const Vector< int, Devices::Host, int >& v );
template long int    VectorOperations< Devices::Host >::getVectorL2Norm( const Vector< long int, Devices::Host, int >& v );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< Devices::Host >::getVectorL2Norm( const Vector< float, Devices::Host, int >& v );
#endif
template double      VectorOperations< Devices::Host >::getVectorL2Norm( const Vector< double, Devices::Host, int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< Devices::Host >::getVectorL2Norm( const Vector< long double, Devices::Host, int >& v );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         VectorOperations< Devices::Host >::getVectorL2Norm( const Vector< int, Devices::Host, long int >& v );
template long int    VectorOperations< Devices::Host >::getVectorL2Norm( const Vector< long int, Devices::Host, long int >& v );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< Devices::Host >::getVectorL2Norm( const Vector< float, Devices::Host, long int >& v );
#endif
template double      VectorOperations< Devices::Host >::getVectorL2Norm( const Vector< double, Devices::Host, long int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< Devices::Host >::getVectorL2Norm( const Vector< long double, Devices::Host, long int >& v );
#endif
#endif


/****
 * Lp norm
 */
template int         VectorOperations< Devices::Host >::getVectorLpNorm( const Vector< int, Devices::Host, int >& v, const int& p );
template long int    VectorOperations< Devices::Host >::getVectorLpNorm( const Vector< long int, Devices::Host, int >& v, const long int& p );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< Devices::Host >::getVectorLpNorm( const Vector< float, Devices::Host, int >& v, const float& p );
#endif
template double      VectorOperations< Devices::Host >::getVectorLpNorm( const Vector< double, Devices::Host, int >& v, const double& p );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< Devices::Host >::getVectorLpNorm( const Vector< long double, Devices::Host, int >& v, const long double& p );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         VectorOperations< Devices::Host >::getVectorLpNorm( const Vector< int, Devices::Host, long int >& v, const int& p );
template long int    VectorOperations< Devices::Host >::getVectorLpNorm( const Vector< long int, Devices::Host, long int >& v, const long int& p );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< Devices::Host >::getVectorLpNorm( const Vector< float, Devices::Host, long int >& v, const float& p );
#endif
template double      VectorOperations< Devices::Host >::getVectorLpNorm( const Vector< double, Devices::Host, long int >& v, const double& p );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< Devices::Host >::getVectorLpNorm( const Vector< long double, Devices::Host, long int >& v, const long double& p );
#endif
#endif



/****
 * Sum
 */
template int         VectorOperations< Devices::Host >::getVectorSum( const Vector< int, Devices::Host, int >& v );
template long int    VectorOperations< Devices::Host >::getVectorSum( const Vector< long int, Devices::Host, int >& v );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< Devices::Host >::getVectorSum( const Vector< float, Devices::Host, int >& v );
#endif
template double      VectorOperations< Devices::Host >::getVectorSum( const Vector< double, Devices::Host, int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< Devices::Host >::getVectorSum( const Vector< long double, Devices::Host, int >& v );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         VectorOperations< Devices::Host >::getVectorSum( const Vector< int, Devices::Host, long int >& v );
template long int    VectorOperations< Devices::Host >::getVectorSum( const Vector< long int, Devices::Host, long int >& v );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< Devices::Host >::getVectorSum( const Vector< float, Devices::Host, long int >& v );
#endif
template double      VectorOperations< Devices::Host >::getVectorSum( const Vector< double, Devices::Host, long int >& v );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< Devices::Host >::getVectorSum( const Vector< long double, Devices::Host, long int >& v );
#endif
#endif

/****
 * Difference max
 */
template int         VectorOperations< Devices::Host >::getVectorDifferenceMax( const Vector< int, Devices::Host, int >& v1, const Vector< int, Devices::Host, int >& v2 );
template long int    VectorOperations< Devices::Host >::getVectorDifferenceMax( const Vector< long int, Devices::Host, int >& v1, const Vector< long int, Devices::Host, int >& v2 );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< Devices::Host >::getVectorDifferenceMax( const Vector< float, Devices::Host, int >& v1,  const Vector< float, Devices::Host, int >& v2);
#endif
template double      VectorOperations< Devices::Host >::getVectorDifferenceMax( const Vector< double, Devices::Host, int >& v1, const Vector< double, Devices::Host, int >& v2 );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< Devices::Host >::getVectorDifferenceMax( const Vector< long double, Devices::Host, int >& v1, const Vector< long double, Devices::Host, int >& v2 );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         VectorOperations< Devices::Host >::getVectorDifferenceMax( const Vector< int, Devices::Host, long int >& v1, const Vector< int, Devices::Host, long int >& v2 );
template long int    VectorOperations< Devices::Host >::getVectorDifferenceMax( const Vector< long int, Devices::Host, long int >& v1, const Vector< long int, Devices::Host, long int >& v2 );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< Devices::Host >::getVectorDifferenceMax( const Vector< float, Devices::Host, long int >& v1, const Vector< float, Devices::Host, long int >& v2 );
#endif
template double      VectorOperations< Devices::Host >::getVectorDifferenceMax( const Vector< double, Devices::Host, long int >& v1, const Vector< double, Devices::Host, long int >& v2 );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< Devices::Host >::getVectorDifferenceMax( const Vector< long double, Devices::Host, long int >& v1, const Vector< long double, Devices::Host, long int >& v2 );
#endif
#endif

/****
 * Difference min
 */
template int         VectorOperations< Devices::Host >::getVectorDifferenceMin( const Vector< int, Devices::Host, int >& v1, const Vector< int, Devices::Host, int >& v2 );
template long int    VectorOperations< Devices::Host >::getVectorDifferenceMin( const Vector< long int, Devices::Host, int >& v1, const Vector< long int, Devices::Host, int >& v2 );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< Devices::Host >::getVectorDifferenceMin( const Vector< float, Devices::Host, int >& v1,  const Vector< float, Devices::Host, int >& v2);
#endif
template double      VectorOperations< Devices::Host >::getVectorDifferenceMin( const Vector< double, Devices::Host, int >& v1, const Vector< double, Devices::Host, int >& v2 );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< Devices::Host >::getVectorDifferenceMin( const Vector< long double, Devices::Host, int >& v1, const Vector< long double, Devices::Host, int >& v2 );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         VectorOperations< Devices::Host >::getVectorDifferenceMin( const Vector< int, Devices::Host, long int >& v1, const Vector< int, Devices::Host, long int >& v2 );
template long int    VectorOperations< Devices::Host >::getVectorDifferenceMin( const Vector< long int, Devices::Host, long int >& v1, const Vector< long int, Devices::Host, long int >& v2 );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< Devices::Host >::getVectorDifferenceMin( const Vector< float, Devices::Host, long int >& v1, const Vector< float, Devices::Host, long int >& v2 );
#endif
template double      VectorOperations< Devices::Host >::getVectorDifferenceMin( const Vector< double, Devices::Host, long int >& v1, const Vector< double, Devices::Host, long int >& v2 );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< Devices::Host >::getVectorDifferenceMin( const Vector< long double, Devices::Host, long int >& v1, const Vector< long double, Devices::Host, long int >& v2 );
#endif
#endif

/****
 * Difference abs max
 */
template int         VectorOperations< Devices::Host >::getVectorDifferenceAbsMax( const Vector< int, Devices::Host, int >& v1, const Vector< int, Devices::Host, int >& v2 );
template long int    VectorOperations< Devices::Host >::getVectorDifferenceAbsMax( const Vector< long int, Devices::Host, int >& v1, const Vector< long int, Devices::Host, int >& v2 );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< Devices::Host >::getVectorDifferenceAbsMax( const Vector< float, Devices::Host, int >& v1,  const Vector< float, Devices::Host, int >& v2);
#endif
template double      VectorOperations< Devices::Host >::getVectorDifferenceAbsMax( const Vector< double, Devices::Host, int >& v1, const Vector< double, Devices::Host, int >& v2 );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< Devices::Host >::getVectorDifferenceAbsMax( const Vector< long double, Devices::Host, int >& v1, const Vector< long double, Devices::Host, int >& v2 );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         VectorOperations< Devices::Host >::getVectorDifferenceAbsMax( const Vector< int, Devices::Host, long int >& v1, const Vector< int, Devices::Host, long int >& v2 );
template long int    VectorOperations< Devices::Host >::getVectorDifferenceAbsMax( const Vector< long int, Devices::Host, long int >& v1, const Vector< long int, Devices::Host, long int >& v2 );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< Devices::Host >::getVectorDifferenceAbsMax( const Vector< float, Devices::Host, long int >& v1, const Vector< float, Devices::Host, long int >& v2 );
#endif
template double      VectorOperations< Devices::Host >::getVectorDifferenceAbsMax( const Vector< double, Devices::Host, long int >& v1, const Vector< double, Devices::Host, long int >& v2 );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< Devices::Host >::getVectorDifferenceAbsMax( const Vector< long double, Devices::Host, long int >& v1, const Vector< long double, Devices::Host, long int >& v2 );
#endif
#endif

/****
 * Difference abs min
 */
template int         VectorOperations< Devices::Host >::getVectorDifferenceAbsMin( const Vector< int, Devices::Host, int >& v1, const Vector< int, Devices::Host, int >& v2 );
template long int    VectorOperations< Devices::Host >::getVectorDifferenceAbsMin( const Vector< long int, Devices::Host, int >& v1, const Vector< long int, Devices::Host, int >& v2 );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< Devices::Host >::getVectorDifferenceAbsMin( const Vector< float, Devices::Host, int >& v1,  const Vector< float, Devices::Host, int >& v2);
#endif
template double      VectorOperations< Devices::Host >::getVectorDifferenceAbsMin( const Vector< double, Devices::Host, int >& v1, const Vector< double, Devices::Host, int >& v2 );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< Devices::Host >::getVectorDifferenceAbsMin( const Vector< long double, Devices::Host, int >& v1, const Vector< long double, Devices::Host, int >& v2 );
#endif

#ifdef INSTANTIATE_LONG_INT
template int         VectorOperations< Devices::Host >::getVectorDifferenceAbsMin( const Vector< int, Devices::Host, long int >& v1, const Vector< int, Devices::Host, long int >& v2 );
template long int    VectorOperations< Devices::Host >::getVectorDifferenceAbsMin( const Vector< long int, Devices::Host, long int >& v1, const Vector< long int, Devices::Host, long int >& v2 );
#ifdef INSTANTIATE_FLOAT
template float       VectorOperations< Devices::Host >::getVectorDifferenceAbsMin( const Vector< float, Devices::Host, long int >& v1, const Vector< float, Devices::Host, long int >& v2 );
#endif
template double      VectorOperations< Devices::Host >::getVectorDifferenceAbsMin( const Vector< double, Devices::Host, long int >& v1, const Vector< double, Devices::Host, long int >& v2 );
#ifdef INSTANTIATE_LONG_DOUBLE
template long double VectorOperations< Devices::Host >::getVectorDifferenceAbsMin( const Vector< long double, Devices::Host, long int >& v1, const Vector< long double, Devices::Host, long int >& v2 );
#endif
#endif


#endif

} // namespace Vectors
} // namespace TNL
