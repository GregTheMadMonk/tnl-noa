#if !defined(_NETGEN_NEUTRAL_H_)
#define _NETGEN_NEUTRAL_H_

#include <fstream>
#include "io_base.h"


namespace implementation
{


template<typename MeshConfigTag>
class NetgenNeutralReader : public IOReader<MeshConfigTag>
{
	friend class Mesh<MeshConfigTag>;

	typedef IOReader<MeshConfigTag>                          IOReaderBase;
	typedef typename IOReaderBase::MeshType                  MeshType;
	typedef typename IOReaderBase::PointType                 PointType;
	typedef typename IOReaderBase::VertexContainerType       VertexContainerType;
	typedef typename IOReaderBase::CellContainerType         CellContainerType;
	typedef typename IOReaderBase::BorderVertexContainerType BorderVertexContainerType;

	enum { cellVerticesCount = IOReaderBase::cellVerticesCount };

public:
	NetgenNeutralReader(const char *filename)
	: m_filename(filename), m_stream(filename)
	{
		if (!m_stream)
			this->error("Unable to open file");
	}

protected:
	virtual void readMesh(MeshType &mesh)
	{
		if (!m_stream)
			this->error("Unable to read from file");

		VertexContainerType &vertexContainer = this->getVertexContainer(mesh);
		CellContainerType   &cellContainer   = this->getCellContainer(mesh);

		std::string line;
		int count;
		std::istringstream iss;

		m_stream >> std::ws;
		std::getline(m_stream, line);
		iss.str(line);
		iss >> count;
		if (!iss || count < 0)
			this->error("Could not read number of vertices, got '", line, "' instead");

		iss.clear();
		vertexContainer.create(count);
		for (int i = 0; i < count; i++)
		{
			PointType point;

			std::getline(m_stream, line);
			iss.str(line);
			for (int d = 0; d < PointType::dimension; d++)
			{
				iss >> point[d];
				if (!iss)
					this->error("Reading vertex coordinates for vertex number ", i, " failed");
			}

			double coord;
			iss >> coord;
			if (iss)
				this->error(PointType::dimension, "-dimensional space expected; more vertex coordinates specified");

			this->setVertexPoint(vertexContainer[i], point);

			iss.clear();
		}

		m_stream >> std::ws;
		std::getline(m_stream, line);
		iss.str(line);
		iss >> count;
		if (!iss || count < 0)
			this->error("Could not read number of cells, got '", line, "' instead");

		iss.clear();
		cellContainer.create(count);
		for (int i = 0; i < count; i++)
		{
			std::getline(m_stream, line);
			iss.str(line);
			int subdomainIndex;
			iss >> subdomainIndex;
			if (!iss)
				this->error("Reading sub-domain index failed for cell number ", i);

			for (int j = 0; j < cellVerticesCount; j++)
			{
				int vertexIndex;
				iss >> vertexIndex;
				if (!iss || vertexIndex < 1)
					this->error("Reading cell vertex index failed for cell number ", i);

				if (vertexIndex > vertexContainer.size())
					this->error("Vertex index greater than total number of vertices for cell number ", i);

				this->setCellVertex(cellContainer[i], j, vertexIndex - 1);
			}

			int vertexIndex;
			iss >> vertexIndex;
			if (iss)
				this->error(cellVerticesCount, " vertex indices for each cell expected, more vertex indices specified");

			iss.clear();
		}
	}

private:
	std::string m_filename;
	std::ifstream m_stream;

	virtual const char *getReaderName() const { return "NetgenNeutralReader"; }
	virtual const char *getFileName() const   { return m_filename.c_str(); }
};


template<typename MeshConfigTag>
class NetgenNeutralWriter : public IOWriter<MeshConfigTag>
{
	friend class Mesh<MeshConfigTag>;

	typedef IOWriter<MeshConfigTag>                          IOWriterBase;
	typedef typename IOWriterBase::MeshType                  MeshType;
	typedef typename IOWriterBase::PointType                 PointType;
	typedef typename IOWriterBase::VertexConstRangeType      VertexConstRangeType;
	typedef typename IOWriterBase::CellConstRangeType        CellConstRangeType;
	typedef typename IOWriterBase::BorderVertexContainerType BorderVertexContainerType;

public:
	NetgenNeutralWriter(const char *filename)
	: m_filename(filename), m_stream(filename)
	{
		if (!m_stream)
			this->error("Unable to open file");
	}

protected:
	void writeMesh(const MeshType &mesh)
	{
		if (!m_stream)
			this->error("Unable to write to file");

		VertexConstRangeType vertexRange = mesh.template entities<0>();
		CellConstRangeType   cellRange   = mesh.template entities<MeshType::dimension>();

		m_stream << vertexRange.size() << "\n";
		for (int i = 0; i < vertexRange.size(); i++)
		{
			const PointType &point = vertexRange[i].getPoint();
			for (DimensionType d = 0; d < PointType::dimension; d++)
				m_stream << point[d] << " ";

			m_stream << "\n";
		}

		m_stream << "\n";
		m_stream << cellRange.size() << "\n";
		for (int i = 0; i < cellRange.size(); i++)
		{
			const BorderVertexContainerType &borderVertexContainer = this->getCellBorderVertexContainer(cellRange[i]);

			m_stream << "1";
			for (int j = 0; j < borderVertexContainer.size(); j++)
				m_stream << " " << borderVertexContainer[j] + 1;

			m_stream << "\n";
		}

		m_stream.flush();

		if (!m_stream)
			this->error("Unable to write to file");
	}

private:
	std::string m_filename;
	std::ofstream m_stream;

	virtual const char *getWriterName() const { return "NetgenNeutralWriter"; }
	virtual const char *getFileName() const   { return m_filename.c_str(); }
};


} // namespace implementation


#endif
