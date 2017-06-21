/***************************************************************************
                          Assert.h  -  description
                             -------------------
    begin                : Jan 12, 2010
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Devices/CudaCallable.h>

/****
 * Debugging assert
 */

#ifndef NDEBUG

#include <sstream>
#include <iostream>
#include <stdio.h>

namespace TNL {
namespace Assert {

inline void
printDiagnosticsHost( const char* assertion,
                      const char* message,
                      const char* file,
                      const char* function,
                      int line,
                      const char* diagnostics )
{
   std::cerr << "Assertion '" << assertion << "' failed !!!\n"
             << "Message: " << message << "\n"
             << "File: " << file << "\n"
             << "Function: " << function << "\n"
             << "Line: " << line << "\n"
             << "Diagnostics:\n" << diagnostics << std::endl;
}

__cuda_callable__
inline void
printDiagnosticsCuda( const char* assertion,
                      const char* message,
                      const char* file,
                      const char* function,
                      int line,
                      const char* diagnostics )
{
   printf( "Assertion '%s' failed !!!\n"
           "Message: %s\n"
           "File: %s\n"
           "Function: %s\n"
           "Line: %d\n"
           "Diagnostics: %s\n",
           assertion, message, file, function, line, diagnostics );
}

__cuda_callable__
inline void
fatalFailure()
{
#ifdef __CUDA_ARCH__
   // https://devtalk.nvidia.com/default/topic/509584/how-to-cancel-a-running-cuda-kernel-/
   // TODO: it is reported as "illegal instruction", but that leads to an abort as well...
   asm("trap;");
#else
   throw EXIT_FAILURE;
#endif
}

template< typename T >
std::string
printToString( const T& value )
{
   ::std::stringstream ss;
   ss << value;
   return ss.str();
}

template<>
inline std::string
printToString( const bool& value )
{
   if( value ) return "true";
   else return "false";
}

template< typename T1, typename T2 >
__cuda_callable__ void
cmpHelperOpFailure( const char* assertion,
                    const char* message,
                    const char* file,
                    const char* function,
                    int line,
                    const char* lhs_expression,
                    const char* rhs_expression,
                    const T1& lhs_value,
                    const T2& rhs_value,
                    const char* op )
{
#ifdef __CUDA_ARCH__
   // diagnostics is not supported - we don't have the machinery
   // to construct the dynamic error message
   printDiagnosticsCuda( assertion, message, file, function, line,
                         "Not supported in CUDA kernels." );
#else
   std::stringstream str;
   if( std::string(op) == "==" ) {
      str << "      Expected: " << lhs_expression;
      if( printToString(lhs_value) != lhs_expression ) {
         str << "\n      Which is: " << lhs_value;
      }
      str << "\nTo be equal to: " << rhs_expression;
      if( printToString(rhs_value) != rhs_expression ) {
         str << "\n      Which is: " << rhs_value;
      }
      str << std::endl;
   }
   else {
      str << "Expected: (" << lhs_expression << ") " << op << " (" << rhs_expression << "), "
          << "actual: " << lhs_value << " vs " << rhs_value << std::endl;
   }
   printDiagnosticsHost( assertion, message, file, function, line,
                         str.str().c_str() );
#endif
   fatalFailure();
}

template< typename T1, typename T2 >
__cuda_callable__ void
cmpHelperTrue( const char* assertion,
               const char* message,
               const char* file,
               const char* function,
               int line,
               const char* expr1,
               const char* expr2,
               const T1& val1,
               const T2& val2 )
{
   // explicit cast is necessary, because T1::operator! might not be defined
   if( ! (bool) val1 )
      ::TNL::Assert::cmpHelperOpFailure( assertion, message, file, function, line,
                                         expr1, "true", val1, true, "==" );
}

template< typename T1, typename T2 >
__cuda_callable__ void
cmpHelperFalse( const char* assertion,
                const char* message,
                const char* file,
                const char* function,
                int line,
                const char* expr1,
                const char* expr2,
                const T1& val1,
                const T2& val2 )
{
   if( val1 )
      ::TNL::Assert::cmpHelperOpFailure( assertion, message, file, function, line,
                                         expr1, "false", val1, false, "==" );
}

// A macro for implementing the helper functions needed to implement
// TNL_ASSERT_??. It is here just to avoid copy-and-paste of similar code.
#define TNL_IMPL_CMP_HELPER_( op_name, op ) \
template< typename T1, typename T2 > \
__cuda_callable__ void \
cmpHelper##op_name( const char* assertion, \
                    const char* message, \
                    const char* file, \
                    const char* function, \
                    int line, \
                    const char* expr1, \
                    const char* expr2, \
                    const T1& val1, \
                    const T2& val2 ) \
{\
   if( ! ( (val1) op (val2) ) ) \
      ::TNL::Assert::cmpHelperOpFailure( assertion, message, file, function, line, \
                                         expr1, expr2, val1, val2, #op );\
}

// Implements the helper function for TNL_ASSERT_EQ
TNL_IMPL_CMP_HELPER_( EQ, == );
// Implements the helper function for TNL_ASSERT_NE
TNL_IMPL_CMP_HELPER_( NE, != );
// Implements the helper function for TNL_ASSERT_LE
TNL_IMPL_CMP_HELPER_( LE, <= );
// Implements the helper function for TNL_ASSERT_LT
TNL_IMPL_CMP_HELPER_( LT, < );
// Implements the helper function for TNL_ASSERT_GE
TNL_IMPL_CMP_HELPER_( GE, >= );
// Implements the helper function for TNL_ASSERT_GT
TNL_IMPL_CMP_HELPER_( GT, > );

#undef TNL_IMPL_CMP_HELPER_

} // namespace Assert
} // namespace TNL

// Internal macro wrapping the __PRETTY_FUNCTION__ "magic".
#if defined( __NVCC__ ) && ( __CUDACC_VER__ < 80000 )
    #define __TNL_PRETTY_FUNCTION "(not known in CUDA 7.5 or older)"
#else
    #define __TNL_PRETTY_FUNCTION __PRETTY_FUNCTION__
#endif

// Internal macro to compose the string representing the assertion.
// We can't do it easily at runtime, because we have to support assertions
// in CUDA kernels, which can't use std::string objects. Instead, we do it
// at compile time - adjacent strings are joined at the language level.
#define __TNL_JOIN_STRINGS( val1, op, val2 ) \
   __STRING( val1 ) " " __STRING( op ) " " __STRING( val2 )

// Internal macro to pass all the arguments to the specified cmpHelperOP
#define __TNL_ASSERT_PRED2( pred, op, val1, val2, msg ) \
   pred( __TNL_JOIN_STRINGS( val1, op, val2 ), \
         msg, __FILE__, __TNL_PRETTY_FUNCTION, __LINE__, \
         #val1, #val2, val1, val2 )
   
// Main definitions of the TNL_ASSERT_* macros
// unary
#define TNL_ASSERT_TRUE( val, msg ) \
   __TNL_ASSERT_PRED2( ::TNL::Assert::cmpHelperTrue, ==, val, true, msg )
#define TNL_ASSERT_FALSE( val, msg ) \
   __TNL_ASSERT_PRED2( ::TNL::Assert::cmpHelperFalse, ==, val, false, msg )
// binary
#define TNL_ASSERT_EQ( val1, val2, msg ) \
   __TNL_ASSERT_PRED2( ::TNL::Assert::cmpHelperEQ, ==, val1, val2, msg )
#define TNL_ASSERT_NE( val1, val2, msg ) \
   __TNL_ASSERT_PRED2( ::TNL::Assert::cmpHelperNE, !=, val1, val2, msg )
#define TNL_ASSERT_LE( val1, val2, msg ) \
   __TNL_ASSERT_PRED2( ::TNL::Assert::cmpHelperLE, <=, val1, val2, msg )
#define TNL_ASSERT_LT( val1, val2, msg ) \
   __TNL_ASSERT_PRED2( ::TNL::Assert::cmpHelperLT, <,  val1, val2, msg )
#define TNL_ASSERT_GE( val1, val2, msg ) \
   __TNL_ASSERT_PRED2( ::TNL::Assert::cmpHelperGE, >=, val1, val2, msg )
#define TNL_ASSERT_GT( val1, val2, msg ) \
   __TNL_ASSERT_PRED2( ::TNL::Assert::cmpHelperGT, >,  val1, val2, msg )




/****
 * Original assert macro with custom command for diagnostic.
 */

// __CUDA_ARCH__ is defined by the compiler only for code executed on GPU
#ifdef __CUDA_ARCH__
#define TNL_ASSERT( ___tnl__assert_condition, ___tnl__assert_command )                                     \
   if( ! ( ___tnl__assert_condition ) )                                                                    \
   {                                                                                                       \
   printf( "Assertion '%s' failed !!! \n File: %s \n Line: %d \n Diagnostics: Not supported with CUDA.\n", \
           __STRING( ___tnl__assert_condition ),                                                           \
           __FILE__,                                                                                       \
           __LINE__ );                                                                                     \
   asm("trap;");                                                                                           \
   }

#else // __CUDA_ARCH__
#define TNL_ASSERT( ___tnl__assert_condition, ___tnl__assert_command )                                  \
   if( ! ( ___tnl__assert_condition ) )                                                                 \
   {                                                                                                    \
   std::cerr << "Assertion '" << __STRING( ___tnl__assert_condition ) << "' failed !!!" << std::endl    \
             << "File: " << __FILE__ << std::endl                                                       \
             << "Function: " << __TNL_PRETTY_FUNCTION << std::endl                                      \
             << "Line: " << __LINE__ << std::endl                                                       \
             << "Diagnostics: ";                                                                        \
        ___tnl__assert_command;                                                                         \
        throw EXIT_FAILURE;                                                                             \
   }
#endif // __CUDA_ARCH__

#else /* #ifndef NDEBUG */

// empty macros for optimized build
#define TNL_ASSERT_TRUE( val, msg )
#define TNL_ASSERT_FALSE( val, msg )
#define TNL_ASSERT_EQ( val1, val2, msg )
#define TNL_ASSERT_NE( val1, val2, msg )
#define TNL_ASSERT_LE( val1, val2, msg )
#define TNL_ASSERT_LT( val1, val2, msg )
#define TNL_ASSERT_GE( val1, val2, msg )
#define TNL_ASSERT_GT( val1, val2, msg )
#define TNL_ASSERT( ___tnl__assert_condition, ___tnl__assert_command )

#endif /* #ifndef NDEBUG */
