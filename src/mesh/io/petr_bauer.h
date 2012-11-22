#if !defined(_PETR_BAUER_H_)
#define _PETR_BAUER_H_

#include <fstream>
#include "io_base.h"


namespace implementation
{


template<typename MeshConfigTag>
class PetrBauerReader : public IOReader<MeshConfigTag>
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
	PetrBauerReader(const char *filename)
	: m_filename(filename), m_stream(filename)
	{
		if (MeshConfigTag::dimWorld > 2 || cellVerticesCount != 3)
			this->error("Petr Bauer format supports only 2D triangular meshes");

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

		std::getline(m_stream, line);
		if (line.compare(0, 10, "level: L ="))
			this->error("Corrupted file header. Line 'level: L =' and not '", line, "' expected");

		std::getline(m_stream, line);
		if (line.compare(0, 10, "nodes: N ="))
			this->error("Corrupted file header. Line 'nodes: N =' and not '", line, "' expected");

		iss.str(line.substr(10));
		iss >> count;
		if (!iss || count < 0)
			this->error("Could not read number of vertices, got '", line.substr(10), "' instead");

		iss.clear();
		vertexContainer.create(count);

		std::getline(m_stream, line);
		if (line.compare(0, 14, "triangles: T ="))
			this->error("Corrupted file header. Line 'triangles: T =' and not '", line, "' expected");

		iss.str(line.substr(14));
		iss >> count;
		if (!iss || count < 0)
			this->error("Could not read number of triangles, got '", line.substr(14), "' instead");

		iss.clear();
		cellContainer.create(count);

		m_stream >> std::ws;
		std::getline(m_stream, line);
		if (line.compare("Nodes:"))
			this->error("Unsupported file format. Vertices header 'Nodes:' and not '", line, "' expected");

		for (int i = 0; i < vertexContainer.size(); i++)
		{
			PointType point;

			std::getline(m_stream, line);
			iss.str(line);

			int index;
			iss >> index;
			if (!iss || index != i)
				this->error("Vertices must be entered in ascending order, which is not satisfied for vertex number ", i);

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
		if (line.compare("Triangles:"))
			this->error("Unsupported file format. Triangles header 'Triangles:' and not '", line, "' expected");

		for (int i = 0; i < cellContainer.size(); i++)
		{
			std::getline(m_stream, line);
			iss.str(line);
			for (int j = 0; j < cellVerticesCount; j++)
			{
				int vertexIndex;
				iss >> vertexIndex;
				if (!iss || vertexIndex < 0)
					this->error("Reading triangle vertex index failed for triangle number ", i);

				if (vertexIndex >= vertexContainer.size())
					this->error("Vertex index greater than total number of vertices for cell number ", i);

				this->setCellVertex(cellContainer[i], j, vertexIndex);
			}

			int vertexIndex;
			iss >> vertexIndex;
			if (iss)
				this->error(cellVerticesCount, " vertex indices for each triangle expected, more vertex indices specified");

			iss.clear();
		}
	}

private:
	std::string m_filename;
	std::ifstream m_stream;

	virtual const char *getReaderName() const { return "PetrBauerReader"; }
	virtual const char *getFileName() const   { return m_filename.c_str(); }
};


template<typename MeshConfigTag>
class PetrBauerWriter : public IOWriter<MeshConfigTag>
{
	friend class Mesh<MeshConfigTag>;

	typedef IOWriter<MeshConfigTag>                          IOWriterBase;
	typedef typename IOWriterBase::MeshType                  MeshType;
	typedef typename IOWriterBase::PointType                 PointType;
	typedef typename IOWriterBase::VertexConstRangeType      VertexConstRangeType;
	typedef typename IOWriterBase::CellConstRangeType        CellConstRangeType;
	typedef typename IOWriterBase::BorderVertexContainerType BorderVertexContainerType;

	enum { cellVerticesCount = IOWriterBase::cellVerticesCount };

public:
	PetrBauerWriter(const char *filename)
	: m_filename(filename), m_stream(filename)
	{
		if (MeshConfigTag::dimWorld > 2 || cellVerticesCount != 3)
			this->error("Petr Bauer format supports only 2D triangular meshes");

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

		m_stream << "level: L = 0\n";
		m_stream << "nodes: N = " << vertexRange.size() << "\n";
		m_stream << "triangles: T = " << cellRange.size() << "\n";
		m_stream << "\n";
		m_stream << "Nodes:\n";
		for (int i = 0; i < vertexRange.size(); i++)
		{
			m_stream << i;

			const PointType &point = vertexRange[i].getPoint();
			for (DimensionType d = 0; d < PointType::dimension; d++)
				m_stream << " " << point[d];

			m_stream << "\n";
		}

		m_stream << "\n";
		m_stream << "Triangles:\n";
		for (int i = 0; i < cellRange.size(); i++)
		{
			const BorderVertexContainerType &borderVertexContainer = this->getCellBorderVertexContainer(cellRange[i]);

			for (int j = 0; j < borderVertexContainer.size(); j++)
				m_stream << borderVertexContainer[j] << " ";

			m_stream << "\n";
		}

		m_stream.flush();

		if (!m_stream)
			this->error("Unable to write to file");
	}

private:
	std::string m_filename;
	std::ofstream m_stream;

	virtual const char *getWriterName() const { return "PetrBauerWriter"; }
	virtual const char *getFileName() const   { return m_filename.c_str(); }
};


} // namespace implementation


#endif
