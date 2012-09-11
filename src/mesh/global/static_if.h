#if !defined(_STATIC_IF_H_)
#define _STATIC_IF_H_


/*
 *     Compile-time if
 *
 * Usage:
 *
 * IF<condition, True>::EXEC(param);
 *
 * It is meant as a substitution to:
 *
 * if (condition)
 *     True::exec(param);
 */
template<bool condition, typename True>
class IF
{
public:
	static void EXEC()
	{
		True::exec();
	}

	template<typename T>
	static void EXEC(T &p)
	{
		True::exec(p);
	}

	template<typename T0, typename T1>
	static void EXEC(T0 &p0, T1 &p1)
	{
		True::exec(p0, p1);
	}

	template<typename T0, typename T1, typename T2>
	static void EXEC(T0 &p0, T1 &p1, T2 &p2)
	{
		True::exec(p0, p1, p2);
	}
};

template<typename True>
class IF<false, True>
{
public:
	static void EXEC() {}

	template<typename T>
	static void EXEC(T &p) {}

	template<typename T0, typename T1>
	static void EXEC(T0 &p0, T1 &p1) {}

	template<typename T0, typename T1, typename T2>
	static void EXEC(T0 &p0, T1 &p1, T2 &p2) {}
};


#endif // !defined(_STATIC_IF_H_)
