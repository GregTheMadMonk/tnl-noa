#if !defined(_VTK_LEGACY_H_)
#define _VTK_LEGACY_H_

#include <fstream>
#include "io_base.h"


namespace implementation
{


class VTKCellTypes
{
protected:
	enum { vtkVertex        =  1,
	       vtkPolyVertex    =  2,
	       vtkLine          =  3,
	       vtkPolyLine      =  4,
	       vtkTriangle      =  5,
	       vtkTriangleStrip =  6,
	       vtkPolygon       =  7,
	       vtkPixel         =  8,
	       vtkQuad          =  9,
	       vtkTetra         = 10,
	       vtkVoxel         = 11,
	       vtkHexahedron    = 12,
	       vtkWedge         = 13,
	       vtkPyramid       = 14};
};


template<typename> struct VTKCellTag {};
template<> struct VTKCellTag<topology::Vertex>        : public VTKCellTypes { enum { number = vtkVertex     }; };
template<> struct VTKCellTag<topology::Edge>          : public VTKCellTypes { enum { number = vtkLine       }; };
template<> struct VTKCellTag<topology::Triangle>      : public VTKCellTypes { enum { number = vtkTriangle   }; };
template<> struct VTKCellTag<topology::Quadrilateral> : public VTKCellTypes { enum { number = vtkQuad       }; };
template<> struct VTKCellTag<topology::Tetrahedron>   : public VTKCellTypes { enum { number = vtkTetra      }; };
template<> struct VTKCellTag<topology::Hexahedron>    : public VTKCellTypes { enum { number = vtkHexahedron }; };


template<typename MeshConfigTag>
class VTKLegacyReader : public IOReader<MeshConfigTag>
{
	friend class Mesh<MeshConfigTag>;

	typedef IOReader<MeshConfigTag>                          IOReaderBase;
	typedef typename IOReaderBase::MeshType                  MeshType;
	typedef typename IOReaderBase::PointType                 PointType;
	typedef typename IOReaderBase::VertexContainerType       VertexContainerType;
	typedef typename IOReaderBase::CellContainerType         CellContainerType;
	typedef typename IOReaderBase::BorderVertexContainerType BorderVertexContainerType;

	enum { cellVerticesCount = IOReaderBase::cellVerticesCount };
	enum { vtkCellType       = VTKCellTag<typename MeshConfigTag::Cell>::number };

public:
	VTKLegacyReader(const char *filename)
	: m_filename(filename), m_stream(filename)
	{
		if (MeshConfigTag::dimWorld > 3)
			this->error("VTK legacy format supports only 1D, 2D and 3D meshes");

		if (!m_stream)
			this->error("Unable to open file");
	}

protected:
	virtual void readMesh(MeshType &mesh)
	{
		if (!m_stream)
			this->error("Unable to read from file");

		VertexContainerType &vertexContainer = getVertexContainer(mesh);
		CellContainerType   &cellContainer   = getCellContainer(mesh);

		std::string line, str1, str2;
		int count, size;
		std::istringstream iss;

		std::getline(m_stream, line);
		if (line.compare(0, 22, "# vtk DataFile Version"))
			this->error("Corrupted file header");

		std::getline(m_stream, line); // Mesh title - unused
		std::getline(m_stream, line);
		if (line.compare("ASCII"))
			this->error("Unsupported file format. Data type 'ASCII' and not '", line, "' expected");

		std::getline(m_stream, line);
		if (line.compare("DATASET UNSTRUCTURED_GRID"))
			this->error("Unsupported file format. Geometry type 'DATASET UNSTRUCTURED_GRID' and not '", line, "' expected");

		m_stream >> std::ws;
		std::getline(m_stream, line);
		iss.str(line);
		iss >> str1 >> count >> str2;
		if (!iss || str1.compare("POINTS") || count < 0 || str2.compare("double"))
			this->error("Unsupported file format. Points header 'POINTS n double' expected and not '", line, "'");

		iss.clear();
		vertexContainer.create(count);
		for (int i = 0; i < count; i++)
		{
			double coord[3];
			m_stream >> coord[0] >> coord[1] >> coord[2];
			if (!m_stream)
				this->error("Reading point coordinates for point number ", i, " failed");

			PointType point;
			for (DimensionType d = 0; d < PointType::dimension; d++)
				point[d] = coord[d];

			for (DimensionType d = PointType::dimension; d < 3; d++)
				if (coord[d] != 0.0)
					this->error(PointType::dimension, "-dimensional mesh expected; other coordinates shold be zero");

			setVertexPoint(vertexContainer[i], point);
		}

		m_stream >> std::ws;
		std::getline(m_stream, line);
		iss.str(line);
		iss >> str1 >> count >> size;
		if (!iss || str1.compare("CELLS") || count < 0 || size < 0)
			this->error("Unsupported file format. Cells header 'CELLS n size' expected and not '", line, "'");

		iss.clear();
		cellContainer.create(count);
		int numbersRead = 0;
		for (int i = 0; i < count; i++)
		{
			int borderVerticesCount;
			m_stream >> borderVerticesCount;
			if (!m_stream || borderVerticesCount != cellVerticesCount)
				this->error("Reading number of cell points failed for cell number ", i);

			for (int j = 0; j < borderVerticesCount; j++)
			{
				int vertexIndex;
				m_stream >> vertexIndex;
				if (!m_stream || vertexIndex < 0)
					this->error("Reading cell points failed for cell number ", i);

				if (vertexIndex >= vertexContainer.size())
					this->error("Point index greater than total number of points for cell number ", i);

				setCellVertex(cellContainer[i], j, vertexIndex);
			}

			numbersRead += (1 + borderVerticesCount);
		}

		if (numbersRead != size)
			this->error("Inconsistent CELLS section - ", size, "numbers promised, but ", numbersRead, " read");

		m_stream >> std::ws;
		std::getline(m_stream, line);
		iss.str(line);
		iss >> str1 >> count;
		if (!iss || str1.compare("CELL_TYPES") || count != cellContainer.size())
			this->error("Unsupported file format. Cell types header 'CELL_TYPES ", cellContainer.size(), "' expected and not '", line, "'");

		for (int i = 0; i < count; i++)
		{
			int cellType;
			m_stream >> cellType;
			if (!m_stream || cellType != vtkCellType)
				this->error("Cell type ", vtkCellType, " expected and not ", cellType);
		}
	}

private:
	std::string m_filename;
	std::ifstream m_stream;

	virtual const char *getReaderName() const { return "VTKLegacyReader"; }
	virtual const char *getFileName() const   { return m_filename.c_str(); }
};


template<typename MeshConfigTag>
class VTKLegacyWriter : public IOWriter<MeshConfigTag>
{
	friend class Mesh<MeshConfigTag>;

	typedef IOWriter<MeshConfigTag>                          IOWriterBase;
	typedef typename IOWriterBase::MeshType                  MeshType;
	typedef typename IOWriterBase::PointType                 PointType;
	typedef typename IOWriterBase::VertexConstRangeType      VertexConstRangeType;
	typedef typename IOWriterBase::CellConstRangeType        CellConstRangeType;
	typedef typename IOWriterBase::BorderVertexContainerType BorderVertexContainerType;

	enum { cellVerticesCount = IOWriterBase::cellVerticesCount };
	enum { vtkCellType       = VTKCellTag<typename MeshConfigTag::Cell>::number };

public:
	VTKLegacyWriter(const char *filename)
	: m_filename(filename), m_stream(filename)
	{
		if (MeshConfigTag::dimWorld > 3)
			this->error("VTK legacy format supports only 1D, 2D and 3D meshes");

		if (!m_stream)
			this->error("Unable to open file");
	}

protected:
	void writeMesh(const MeshType &mesh)
	{
		if (!m_stream)
			this->error("Unable to write to file");

		VertexConstRangeType vertexRange = mesh.template entities<0>();
		CellConstRangeType   cellRange   = mesh.template entities<MeshConfigTag::dimension>();

		m_stream << "# vtk DataFile Version 2.0\n";
		m_stream << "Mesh\n";
		m_stream << "ASCII\n";
		m_stream << "DATASET UNSTRUCTURED_GRID\n";
		m_stream << "\n";
		m_stream << "POINTS " << vertexRange.size() << " double\n";
		for (int i = 0; i < vertexRange.size(); i++)
		{
			const PointType &point = vertexRange[i].getPoint();
			for (DimensionType d = 0; d < PointType::dimension; d++)
				m_stream << point[d] << " ";

			for (DimensionType d = PointType::dimension; d < 3; d++)
				m_stream << "0 ";

			m_stream << "\n";
		}

		m_stream << "\n";
		m_stream << "CELLS " << cellRange.size() << " " << cellRange.size()*(1 + cellVerticesCount) << "\n";
		for (int i = 0; i < cellRange.size(); i++)
		{
			const BorderVertexContainerType &borderVertexContainer = getCellBorderVertexContainer(cellRange[i]);
			assert(borderVertexContainer.size() == cellVerticesCount);

			m_stream << cellVerticesCount;
			for (int j = 0; j < cellVerticesCount; j++)
				m_stream << " " << borderVertexContainer[j];

			m_stream << "\n";
		}

		m_stream << "\n";
		m_stream << "CELL_TYPES " << cellRange.size() << "\n";
		for (int i = 0; i < cellRange.size(); i++)
			m_stream << vtkCellType << "\n";

		m_stream.flush();

		if (!m_stream)
			this->error("Unable to write to file");
	}

private:
	std::string m_filename;
	std::ofstream m_stream;

	virtual const char *getWriterName() const { return "VTKLegacyWriter"; }
	virtual const char *getFileName() const   { return m_filename.c_str(); }
};


} // namespace implementation


#endif
