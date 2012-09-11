#if !defined(_COMMON_H_)
#define _COMMON_H_


namespace implementation
{


typedef int DimensionType;


template<DimensionType dimension>
class DimTag
{
public:
	enum { value = dimension };

	typedef DimTag<dimension - 1> Previous;
};


// Tags denoting whether certain mesh entities (by context) should or should not be stored
template<bool storageEnabled>
class StorageTag
{
public:
	enum { enabled = storageEnabled };
};


} // namespace implementation


using implementation::DimensionType;


#endif
