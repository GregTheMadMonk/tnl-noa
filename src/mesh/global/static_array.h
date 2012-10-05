#if !defined(_STATIC_ARRAY_H_)
#define _STATIC_ARRAY_H_

#include <cassert>
#include <algorithm>


template<typename T, typename I, I size>
class StaticArray
{
public:
	void setLike(const StaticArray<T, I, size> &array);

	I getSize() const;

	const T &operator[](I index) const;
	T &operator[](I index);

	void sort();

private:
	T m_data[size];
};


template<typename T, typename I, I size>
void StaticArray<T, I, size> :: setLike( const StaticArray<T, I, size> &array )
{
	for (I i = 0; i < size; i++)
		(*this)[i] = array[i];
}

template<typename T, typename I, I size>
inline I StaticArray<T, I, size>::getSize() const
{
	return size;
}

template<typename T, typename I, I size>
inline const T &StaticArray<T, I, size>::operator[](I index) const
{
	assert(index >= 0 && index < size);

	return m_data[index];
}

template<typename T, typename I, I size>
inline T &StaticArray<T, I, size>::operator[](I index)
{
	assert(index >= 0 && index < size);

	return m_data[index];
}

template<typename T, typename I, I size>
inline void StaticArray<T, I, size>::sort()
{
	std::sort(m_data, m_data + getSize());
}


#endif // !defined(_STATIC_ARRAY_H_)
