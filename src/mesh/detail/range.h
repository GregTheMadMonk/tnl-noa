#if !defined(_RANGE_H_)
#define _RANGE_H_


namespace implementation
{


template<typename ContainerType> class ConstRange;
template<typename DataContainerType, typename IndexContainerType> class ConstIndirectRange;


template<typename ContainerType>
class Range
{
	friend class ConstRange<ContainerType>;

public:
	typedef typename ContainerType::DataType  DataType;
	typedef typename ContainerType::IndexType IndexType;

	explicit Range(ContainerType &container)       : m_container(&container) {}

	IndexType size() const                         { return m_container->size(); }

	DataType &operator[](IndexType id)             { assert(0 <= id && id < size()); return (*m_container)[id]; }
	const DataType &operator[](IndexType id) const { assert(0 <= id && id < size()); return (*m_container)[id]; }

private:
	ContainerType *m_container;
};

template<typename ContainerType>
class ConstRange
{
	typedef Range<ContainerType>              RangeType;

public:
	typedef typename ContainerType::DataType  DataType;
	typedef typename ContainerType::IndexType IndexType;

	ConstRange(const RangeType &range)                  : m_container(range.m_container) {}
	explicit ConstRange(const ContainerType &container) : m_container(&container) {}

	IndexType size() const                              { return m_container->size(); }

	const DataType &operator[](IndexType id) const      { assert(0 <= id && id < size()); return (*m_container)[id]; }

	ConstRange &operator=(const RangeType &range)       { m_container = range.m_container; return *this; }

private:
	const ContainerType *m_container;
};


template<typename DataRangeType, typename IndexContainerType>
class IndirectRange
{
	friend class ConstIndirectRange<DataRangeType, IndexContainerType>;

public:
	typedef typename DataRangeType::DataType       DataType;
	typedef typename IndexContainerType::IndexType IndexType;

	IndirectRange(const DataRangeType &dataRange, IndexContainerType &indexContainer)
		: m_dataRange(dataRange), m_indexContainer(&indexContainer) {}

	IndexType size() const                         { return m_indexContainer->size(); }

	DataType &operator[](IndexType id)             { assert(0 <= id && id < size()); return m_dataRange[(*m_indexContainer)[id]]; }
	const DataType &operator[](IndexType id) const { assert(0 <= id && id < size()); return m_dataRange[(*m_indexContainer)[id]]; }

private:
	DataRangeType m_dataRange;
	IndexContainerType *m_indexContainer;
};

template<typename DataRangeType, typename IndexContainerType>
class ConstIndirectRange
{
	typedef IndirectRange<DataRangeType, IndexContainerType> IndirectRangeType;

public:
	typedef typename DataRangeType::DataType                 DataType;
	typedef typename IndexContainerType::IndexType           IndexType;

	ConstIndirectRange(const IndirectRangeType &range)
		: m_dataRange(range.m_dataRange), m_indexContainer(range.m_indexContainer) {}

	ConstIndirectRange(const DataRangeType &dataRange, const IndexContainerType &indexContainer)
		: m_dataRange(dataRange), m_indexContainer(&indexContainer) {}

	IndexType size() const                         { return m_indexContainer->size(); }

	const DataType &operator[](IndexType id) const { assert(0 <= id && id < size()); return m_dataRange[(*m_indexContainer)[id]]; }

	ConstIndirectRange &operator=(const IndirectRangeType &range)
	{
		m_dataRange = range.m_dataRange;
		m_indexContainer = range.m_indexContainer;
		return *this;
	}

private:
	const DataRangeType m_dataRange;
	const IndexContainerType *m_indexContainer;
};


} // namespace implementation


#endif
