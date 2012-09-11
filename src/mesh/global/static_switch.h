#if !defined(_STATIC_SWITCH_H_)
#define _STATIC_SWITCH_H_


/*
 *     Compile-time switch with return
 *
 * Usage:
 *
 * SWITCH_RETURN<ReturnType, 5, Case, Default>::EXEC(n, param);
 *
 * It is meant as a substitution to:
 *
 * switch (n)
 * {
 *     case 0:  return Case<0>::exec(param);
 *     case 1:  return Case<1>::exec(param);
 *     case 2:  return Case<2>::exec(param);
 *     case 3:  return Case<3>::exec(param);
 *     case 4:  return Case<4>::exec(param);
 *     default: return Default::exec(n, param);
 * }
 */
template<typename ReturnType, int N, template<int> class Case, typename Default>
class SWITCH_RETURN
{
public:
	static ReturnType EXEC(int n)
	{
		if (n == N - 1)
			return Case<N - 1>::exec();
		else
			return SWITCH_RETURN<ReturnType, N - 1, Case, Default>::EXEC(n);
	}

	template<typename T>
	static ReturnType EXEC(int n, T &p)
	{
		if (n == N - 1)
			return Case<N - 1>::exec(p);
		else
			return SWITCH_RETURN<ReturnType, N - 1, Case, Default>::EXEC(n, p);
	}

	template<typename T0, typename T1>
	static ReturnType EXEC(int n, T0 &p0, T1 &p1)
	{
		if (n == N - 1)
			return Case<N - 1>::exec(p0, p1);
		else
			return SWITCH_RETURN<ReturnType, N - 1, Case, Default>::EXEC(n, p0, p1);
	}

	template<typename T0, typename T1, typename T2>
	static ReturnType EXEC(int n, T0 &p0, T1 &p1, T2 &p2)
	{
		if (n == N - 1)
			return Case<N - 1>::exec(p0, p1, p2);
		else
			return SWITCH_RETURN<ReturnType, N - 1, Case, Default>::EXEC(n, p0, p1, p2);
	}
};

template<typename ReturnType, template<int> class Case, typename Default>
class SWITCH_RETURN<ReturnType, 0, Case, Default>
{
public:
	static ReturnType EXEC(int n)
	{
		return Default::exec(n);
	}

	template<typename T>
	static ReturnType EXEC(int n, T &p)
	{
		return Default::exec(n, p);
	}

	template<typename T0, typename T1>
	static ReturnType EXEC(int n, T0 &p0, T1 &p1)
	{
		return Default::exec(n, p0, p1);
	}

	template<typename T0, typename T1, typename T2>
	static ReturnType EXEC(int n, T0 &p0, T1 &p1, T2 &p2)
	{
		return Default::exec(n, p0, p1, p2);
	}
};


#endif // !defined(_STATIC_SWITCH_H_)
