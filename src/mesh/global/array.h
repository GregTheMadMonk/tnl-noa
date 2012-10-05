#if !defined(_ARRAY_H_)
#define _ARRAY_H_

#include <cassert>

#include <iostream>

using namespace std;

template<typename T, typename I = int>
class Array
{
public:
	Array();
	explicit Array(I size);
	Array(const Array<T, I> &array);
	~Array();

	void setSize(I size);
	void setLike(const Array<T, I> &array);
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

	setSize(size);
}

template<typename T, typename I>
Array<T, I>::Array(const Array<T, I> &array)
: m_data(0), m_size(0)
{
	setLike(array);
}

template<typename T, typename I>
Array<T, I>::~Array()
{
	free();
}

template<typename T, typename I>
void Array<T, I>:: setSize(I size)
{
	assert(size >= 0);

	if (m_data && getSize() == size)
	{
	   cerr << "Not-allocating " << this -> m_size << " data at " << this -> m_data << endl;
		return;
	}

	free();
	if (size > 0)
	{
		m_data = new T[size];
		m_size = size;
		cerr << "Allocating " << this -> m_size << " data at " << this -> m_data << endl;
	}

}

template<typename T, typename I>
void Array<T, I>:: setLike(const Array<T, I> &array)
{
	setSize(array.getSize());

#pragma omp parallel for schedule(static)
	for (I i = 0; i < array.getSize(); i++)
		(*this)[i] = array[i];
}

template<typename T, typename I>
void Array<T, I>::free()
{
	if( m_data )
	{
	   cerr << "Freeing data at " << this -> m_data << endl;
	   delete[] m_data;
	}
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
		setLike(array);

	return *this;
}


#endif // !defined(_ARRAY_H_)
