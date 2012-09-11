#if !defined(_ARRAY_H_)
#define _ARRAY_H_

#include <cassert>


template<typename T, typename I = int>
class Array
{
public:
	Array();
	explicit Array(I size);
	Array(const Array<T, I> &array);
	~Array();

	void create(I size);
	void create(const Array<T, I> &array);
	void free();

	I getSize() const;

	const T &operator[](I index) const;
	T &operator[](I index);

	Array &operator=(const Array<T, I> &array);

private:
	T *m_data;
	I m_size;
};


template<typename T, typename I>
Array<T, I>::Array()
: m_data(0), m_size(0)
{
}

template<typename T, typename I>
Array<T, I>::Array(I size)
: m_data(0), m_size(0)
{
	assert(size >= 0);

	create(size);
}

template<typename T, typename I>
Array<T, I>::Array(const Array<T, I> &array)
: m_data(0), m_size(0)
{
	create(array);
}

template<typename T, typename I>
Array<T, I>::~Array()
{
	free();
}

template<typename T, typename I>
void Array<T, I>::create(I size)
{
	assert(size >= 0);

	if (m_data && getSize() == size)
		return;

	free();
	if (size > 0)
	{
		m_data = new T[size];
		m_size = size;
	}
}

template<typename T, typename I>
void Array<T, I>::create(const Array<T, I> &array)
{
	create(array.getSize());

#pragma omp parallel for schedule(static)
	for (I i = 0; i < array.getSize(); i++)
		(*this)[i] = array[i];
}

template<typename T, typename I>
void Array<T, I>::free()
{
	delete[] m_data;
	m_data = 0;
	m_size = 0;
}

template<typename T, typename I>
inline I Array<T, I>::getSize() const
{
	return m_size;
}

template<typename T, typename I>
inline const T &Array<T, I>::operator[](I index) const
{
	assert(index >= 0 && index < getSize());

	return m_data[index];
}

template<typename T, typename I>
inline T &Array<T, I>::operator[](I index)
{
	assert(index >= 0 && index < getSize());

	return m_data[index];
}

template<typename T, typename I>
Array<T, I> &Array<T, I>::operator=(const Array<T, I> &array)
{
	if (this != &array)
		create(array);

	return *this;
}


#endif // !defined(_ARRAY_H_)
