#if !defined(_STATIC_ASSERT_H_)
#define _STATIC_ASSERT_H_


// static_assert() available for g++ 4.3 or newer with -std=c++0x or -std=gnu++0x
#if (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ > 2)) && defined(__GXX_EXPERIMENTAL_CXX0X__)
#define CXX0X_STATIC_ASSERT_AVAILABLE
#endif


#if defined(CXX0X_STATIC_ASSERT_AVAILABLE)
#define STATIC_ASSERT(expression, msg) static_assert(expression, msg)
#else

#define JOIN(X, Y)  JOIN2(X, Y)
#define JOIN2(X, Y) X##Y

// Incomplete-type implementation of compile time assertion
namespace static_assert_impl
{
template<bool x> struct STATIC_ASSERTION_FAILURE;
template<>       struct STATIC_ASSERTION_FAILURE<true> { enum { value = 1 }; };

template<int x> struct static_assert_test{};
}

#define STATIC_ASSERT(expression, msg) \
	typedef static_assert_impl::static_assert_test<sizeof(static_assert_impl::STATIC_ASSERTION_FAILURE<expression>)>\
		JOIN(static_assertion_failure_identifier, __LINE__)

#endif // defined(CXX0X_STATIC_ASSERT_AVAILABLE)


#endif // !defined(_STATIC_ASSERT_H_)
