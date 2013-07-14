#if !defined(_CONTAINER_H_)
#define _CONTAINER_H_

#include <map>
#include <list>
#include <stdexcept>
#include <mesh/global/array.h>
#include <core/arrays/tnlArray.h>
#include <mesh/global/static_array.h>


namespace implementation
{


template<typename T, typename I>
class Container
{
public:
	typedef T DataType;
	typedef I IndexType;

	Container()                                    : m_data( "mesh-container" ) {}
	explicit Container(IndexType size)             : m_data( "mesh-container", size) { assert(0 <= size); }

	void create(IndexType size)                    { assert(0 <= size); m_data. setSize(size); }
	void free()                                    { m_data. reset() /*free()*/; }

	IndexType size() const                         { return m_data.getSize(); }

	DataType &operator[](IndexType id)             { assert(0 <= id && id < size()); return m_data[id]; }
	const DataType &operator[](IndexType id) const { assert(0 <= id && id < size()); return m_data[id]; }

	size_t allocatedMemorySize() const             { return sizeof(DataType)*size(); }

private:
	tnlArray< DataType, tnlHost, IndexType > m_data;
	//Array< DataType, IndexType > m_data;
};


template<typename T, typename I, I numElements>
class StaticContainer
{
public:
	typedef T DataType;
	typedef I IndexType;

	IndexType size() const                         { return m_data.getSize(); }

	DataType &operator[](IndexType id)             { assert(0 <= id && id < size()); return m_data[id]; }
	const DataType &operator[](IndexType id) const { assert(0 <= id && id < size()); return m_data[id]; }

	void sort()                                    { m_data.sort(); }

	size_t allocatedMemorySize() const             { return 0; }

private:
	StaticArray<DataType, IndexType, numElements> m_data;
};


// Unique container (map) - stores each object at most once.
// Uses operator < to compare objects by their keys.
template<typename T, typename Key>
class UniqueContainer
{
	typedef std::map<Key, T>                 MapType;
	typedef typename MapType::value_type     MapValueType;
	typedef typename MapType::const_iterator MapConstIteratorType;

public:
	typedef T   DataType;
	typedef Key KeyType;

	class ConstIterator
	{
		friend class UniqueContainer;

	public:
		ConstIterator()                                  {}

		const T &operator*() const                       { return m_iterator->second; }
		const T *operator->() const                      { return &(operator*()); }

		ConstIterator &operator++()                      { ++m_iterator; return *this; }
		const ConstIterator operator++(int)              { ConstIterator iter(*this); ++(*this); return iter; }

		bool operator==(const ConstIterator &iter) const { return (m_iterator == iter.m_iterator); }
		bool operator!=(const ConstIterator &iter) const { return !(*this == iter); }

	private:
		MapConstIteratorType m_iterator;

		explicit ConstIterator(const MapConstIteratorType &iter) : m_iterator(iter) {}
	};

	ConstIterator insert(const DataType &data)
	{
		MapValueType value(KeyType(data), data);
		std::pair<MapConstIteratorType, bool> ret = m_map.insert(value);

		return ConstIterator(ret.first);
	}

	ConstIterator find(const DataType &data) const
	{
		MapConstIteratorType iter = m_map.find(KeyType(data));
		if (iter == m_map.end())
			return end();

		return ConstIterator(iter);
	}

	size_t size() const         { return m_map.size(); }
	void free()                 { m_map.clear(); }

	ConstIterator begin() const { return ConstIterator(m_map.begin()); }
	ConstIterator end() const   { return ConstIterator(m_map.end()); }

private:
	MapType m_map;
};


// Index is assigned to each object added to the container.
template<typename T, typename I, typename Key>
class IndexedUniqueContainer
{
	struct DataWithIndex
	{
		explicit DataWithIndex(const T &data) : m_data(data) {}
		DataWithIndex(const T &data, I index) : m_data(data), m_index(index) {}

		T m_data;
		I m_index;
	};

	struct KeyWithIndex : public Key
	{
		KeyWithIndex(const DataWithIndex &data) : Key(data.m_data) {}
	};

	typedef UniqueContainer<DataWithIndex, KeyWithIndex> UniqueContainerType;
	typedef typename UniqueContainerType::ConstIterator  UniqueContainerIterator;

public:
	typedef T   DataType;
	typedef I   IndexType;
	typedef Key KeyType;

	class Exception : public std::runtime_error
	{
	public:
		explicit Exception(const std::string &message) : std::runtime_error(message) {}
	};

	IndexType insert(const DataType &data)
	{
		DataWithIndex newData(data, m_uniqueContainer.size());
		UniqueContainerIterator iter = m_uniqueContainer.insert(newData);
		assert(iter != m_uniqueContainer.end());

		return iter->m_index;
	}

	IndexType find(const DataType &data) const
	{
		UniqueContainerIterator iter = m_uniqueContainer.find(DataWithIndex(data));
		if (iter == m_uniqueContainer.end())
			throw Exception("IndexedUniqueContainer error: find() method - data not found");

		return iter->m_index;
	}

	void copy(Container<DataType, IndexType> &container) const
	{
		container.create(m_uniqueContainer.size());
		for (UniqueContainerIterator iter = m_uniqueContainer.begin(); iter != m_uniqueContainer.end(); ++iter)
			container[iter->m_index] = iter->m_data;
	}

	size_t size() const { return m_uniqueContainer.size(); }
	void free()         { m_uniqueContainer.free(); }

private:
	UniqueContainerType m_uniqueContainer;
};


// Container with fast insertion of elements
template<typename T>
class GrowableContainer
{
	typedef std::list<T>                      ListType;
	typedef typename ListType::const_iterator ListConstIteratorType;

public:
	typedef T DataType;

	void insert(const DataType &data) { m_data.push_back(data); }

	size_t size() const               { return m_data.size(); }

	void free()                       { m_data.clear(); }

	template<typename IndexType>
	void copy(Container<DataType, IndexType> &container) const
	{
		IndexType i = 0;
		container.create(size());
		for (ListConstIteratorType iter = m_data.begin(); iter != m_data.end(); ++iter)
			container[i++] = *iter;
	}

private:
	std::list<DataType> m_data;
};


} // namespace implementation


#endif
