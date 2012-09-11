#if !defined(_STATIC_LOOP_H_)
#define _STATIC_LOOP_H_


/*
 *     Compile-time loop
 *
 * Usage:
 *
 * LOOP<int, 5, LoopBody>::EXEC(param);
 *
 * It is meant as a substitution to:
 *
 * for (int i = 0; i < 5; i++)
 *     LoopBody<i>::exec(param);
 */
namespace static_loop_impl
{

template<typename IndexType, IndexType val>
class IndexTag
{
public:
	static const IndexType value = val;

	typedef IndexTag<IndexType, val - 1> Decrement;
};


template<typename IndexType, typename N, template<IndexType> class LoopBody>
class StaticLoop
{
public:
	static void exec()
	{
		StaticLoop<IndexType, typename N::Decrement, LoopBody>::exec();
		LoopBody<N::value - 1>::exec();
	}

	template<typename T>
	static void exec(T &p)
	{
		StaticLoop<IndexType, typename N::Decrement, LoopBody>::exec(p);
		LoopBody<N::value - 1>::exec(p);
	}

	template<typename T0, typename T1>
	static void exec(T0 &p0, T1 &p1)
	{
		StaticLoop<IndexType, typename N::Decrement, LoopBody>::exec(p0, p1);
		LoopBody<N::value - 1>::exec(p0, p1);
	}

	template<typename T0, typename T1, typename T2>
	static void exec(T0 &p0, T1 &p1, T2 &p2)
	{
		StaticLoop<IndexType, typename N::Decrement, LoopBody>::exec(p0, p1, p2);
		LoopBody<N::value - 1>::exec(p0, p1, p2);
	}
};

template<typename IndexType, template<IndexType> class LoopBody>
class StaticLoop<IndexType, IndexTag<IndexType, 0>, LoopBody>
{
public:
	static void exec() {}

	template<typename T>
	static void exec(T &p) {}

	template<typename T0, typename T1>
	static void exec(T0 &p0, T1 &p1) {}

	template<typename T0, typename T1, typename T2>
	static void exec(T0 &p0, T1 &p1, T2 &p2) {}
};

} // namespace static_loop_impl


template<typename IndexType, IndexType N, template<IndexType> class LoopBody>
class LOOP
{
public:
	static void EXEC()
	{
		static_loop_impl::StaticLoop<IndexType, static_loop_impl::IndexTag<IndexType, N>, LoopBody>::exec();
	}

	template<typename T>
	static void EXEC(T &p)
	{
		static_loop_impl::StaticLoop<IndexType, static_loop_impl::IndexTag<IndexType, N>, LoopBody>::exec(p);
	}

	template<typename T0, typename T1>
	static void EXEC(T0 &p0, T1 &p1)
	{
		static_loop_impl::StaticLoop<IndexType, static_loop_impl::IndexTag<IndexType, N>, LoopBody>::exec(p0, p1);
	}

	template<typename T0, typename T1, typename T2>
	static void EXEC(T0 &p0, T1 &p1, T2 &p2)
	{
		static_loop_impl::StaticLoop<IndexType, static_loop_impl::IndexTag<IndexType, N>, LoopBody>::exec(p0, p1, p2);
	}
};


#endif // !defined(_STATIC_LOOP_H_)
