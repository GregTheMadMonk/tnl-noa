#if !defined(_IO_BASE_H_)
#define _IO_BASE_H_

#include <sstream>
#include <mesh/detail/mesh_tags.h>


namespace implementation
{


template<typename> class Mesh;


class IOBase
{
public:
	class IOException : public std::runtime_error
	{
	public:
		explicit IOException(const std::string &message) : std::runtime_error(message) {}
	};

	virtual ~IOBase() {}

protected:
	template<typename T0>
	void error(const T0 &msg0) const
	{
		std::ostringstream oss;
		oss << getErrorContext() << msg0;
		throwException(oss.str());
	}

	template<typename T0, typename T1>
	void error(const T0 &msg0, const T1 &msg1) const
	{
		std::ostringstream oss;
		oss << getErrorContext() << msg0 << msg1;
		throwException(oss.str());
	}

	template<typename T0, typename T1, typename T2>
	void error(const T0 &msg0, const T1 &msg1, const T2 &msg2) const
	{
		std::ostringstream oss;
		oss << getErrorContext() << msg0 << msg1 << msg2;
		throwException(oss.str());
	}

	template<typename T0, typename T1, typename T2, typename T3>
	void error(const T0 &msg0, const T1 &msg1, const T2 &msg2, const T3 &msg3) const
	{
		std::ostringstream oss;
		oss << getErrorContext() << msg0 << msg1 << msg2 << msg3;
		throwException(oss.str());
	}

	template<typename T0, typename T1, typename T2, typename T3, typename T4>
	void error(const T0 &msg0, const T1 &msg1, const T2 &msg2, const T3 &msg3, const T4 &msg4) const
	{
		std::ostringstream oss;
		oss << getErrorContext() << msg0 << msg1 << msg2 << msg3 << msg4;
		throwException(oss.str());
	}

private:
	static void throwException(const std::string &message)
	{
		throw IOException(message);
	}

	virtual std::string getErrorContext() const = 0;
};


template<typename MeshConfigTag>
class IOReader : public IOBase
{
public:
	typedef Mesh<MeshConfigTag> MeshType;

	virtual void readMesh(MeshType &mesh) = 0;

protected:
	typedef typename MeshTag<MeshConfigTag>::PointType PointType;

	typedef typename EntityTag<MeshConfigTag, DimTag<0>                   >::Type          VertexType;
	typedef typename EntityTag<MeshConfigTag, DimTag<0>                   >::ContainerType VertexContainerType;
	typedef typename EntityTag<MeshConfigTag, DimTag<MeshType::dimension> >::Type          CellType;
	typedef typename EntityTag<MeshConfigTag, DimTag<MeshType::dimension> >::ContainerType CellContainerType;

	typedef typename EntityBorderTag<MeshConfigTag, typename MeshConfigTag::Cell, DimTag<0> >::ContainerType BorderVertexContainerType;
	typedef typename BorderVertexContainerType::DataType                                                     GlobalIndexType;
	typedef typename BorderVertexContainerType::IndexType                                                    LocalIndexType;

	enum { cellVerticesCount = EntityBorderTag<MeshConfigTag, typename MeshConfigTag::Cell, DimTag<0> >::count };

	VertexContainerType &getVertexContainer(MeshType &mesh) const                                    { return mesh.entityContainer(DimTag<0>()); }
	CellContainerType &getCellContainer(MeshType &mesh) const                                        { return mesh.entityContainer(DimTag<MeshType::dimension>()); }

	void setVertexPoint(VertexType &vertex, const PointType &point) const                            { vertex.setPoint(point); }
	void setCellVertex(CellType &cell, LocalIndexType localIndex, GlobalIndexType globalIndex) const { cell.setVertex(localIndex, globalIndex); }

private:
	virtual const char *getReaderName() const = 0;
	virtual const char *getFileName() const = 0;

	virtual std::string getErrorContext() const
	{
		std::ostringstream oss;
		oss << getReaderName() << " error reading file '" << getFileName() << "': ";
		return oss.str();
	}
};


template<typename MeshConfigTag>
class IOWriter : public IOBase
{
public:
	typedef Mesh<MeshConfigTag> MeshType;

	virtual void writeMesh(const MeshType &mesh) = 0;

protected:
	typedef typename MeshTag<MeshConfigTag>::PointType PointType;

	typedef typename EntityTag<MeshConfigTag, DimTag<0>                   >::Type           VertexType;
	typedef typename EntityTag<MeshConfigTag, DimTag<0>                   >::ConstRangeType VertexConstRangeType;
	typedef typename EntityTag<MeshConfigTag, DimTag<MeshType::dimension> >::Type           CellType;
	typedef typename EntityTag<MeshConfigTag, DimTag<MeshType::dimension> >::ConstRangeType CellConstRangeType;

	typedef typename EntityBorderTag<MeshConfigTag, typename MeshConfigTag::Cell, DimTag<0> >::ContainerType BorderVertexContainerType;

	enum { cellVerticesCount = EntityBorderTag<MeshConfigTag, typename MeshConfigTag::Cell, DimTag<0> >::count };

	const BorderVertexContainerType &getCellBorderVertexContainer(const CellType &cell) const { return cell.borderContainer(DimTag<0>()); }

private:
	virtual const char *getWriterName() const = 0;
	virtual const char *getFileName() const = 0;

	virtual std::string getErrorContext() const
	{
		std::ostringstream oss;
		oss << getWriterName() << " error writing file '" << getFileName() << "': ";
		return oss.str();
	}
};


} // namespace implementation


#endif
