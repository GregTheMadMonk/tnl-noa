#if !defined(_MESH_IO_H_)
#define _MESH_IO_H_

#include <memory>
#include <mesh/global/static_switch.h>
#include <mesh/io/netgen_neutral.h>
#include <mesh/io/petr_bauer.h>
#include <mesh/io/vtk_legacy.h>


namespace implementation
{


template<typename MeshConfigTag, int index> struct IOReaders {};
template<typename MeshConfigTag> struct IOReaders<MeshConfigTag, 0> { typedef VTKLegacyReader<MeshConfigTag>     Type; };
template<typename MeshConfigTag> struct IOReaders<MeshConfigTag, 1> { typedef NetgenNeutralReader<MeshConfigTag> Type; };
template<typename MeshConfigTag> struct IOReaders<MeshConfigTag, 2> { typedef PetrBauerReader<MeshConfigTag>     Type; };


static const char *readersExtensions[] = {
	"vtk",  // 0 - VTKLegacyReader
	"mesh", // 1 - NetgenNeutralReader
	"tri"   // 2 - PetrBauerReader
};


template<typename MeshConfigTag, int index> struct IOWriters {};
template<typename MeshConfigTag> struct IOWriters<MeshConfigTag, 0> { typedef VTKLegacyWriter<MeshConfigTag>     Type; };
template<typename MeshConfigTag> struct IOWriters<MeshConfigTag, 1> { typedef NetgenNeutralWriter<MeshConfigTag> Type; };
template<typename MeshConfigTag> struct IOWriters<MeshConfigTag, 2> { typedef PetrBauerWriter<MeshConfigTag>     Type; };


static const char *writersExtensions[] = {
	"vtk",  // 0 - VTKLegacyWriter
	"mesh", // 1 - NetgenNeutralWriter
	"tri"   // 2 - PetrBauerReader
};


template<typename MeshConfigTag>
class IOFactory
{
public:
	typedef IOReader<MeshConfigTag> ReaderType;
	typedef IOWriter<MeshConfigTag> WriterType;

	typedef std::auto_ptr<ReaderType> ReaderAutoPtr;
	typedef std::auto_ptr<WriterType> WriterAutoPtr;

private:
	template<int N>
	struct IOReaderSelector
	{
		static ReaderAutoPtr exec(const char *filename) { return ReaderAutoPtr(new typename IOReaders<MeshConfigTag, N>::Type(filename)); }
	};

	template<int N>
	struct IOWriterSelector
	{
		static WriterAutoPtr exec(const char *filename) { return WriterAutoPtr(new typename IOWriters<MeshConfigTag, N>::Type(filename)); }
	};

	struct IOReaderDefault
	{
		static ReaderAutoPtr exec(int n, const char *filename) { throw std::runtime_error("Error: Unknown mesh reader"); }
	};

	struct IOWriterDefault
	{
		static WriterAutoPtr exec(int n, const char *filename) { throw std::runtime_error("Error: Unknown mesh writer"); }
	};

	static const int readersCount = sizeof(readersExtensions)/sizeof(readersExtensions[0]);
	static const int writersCount = sizeof(writersExtensions)/sizeof(writersExtensions[0]);

public:
	static ReaderAutoPtr getReader(const char *filename)
	{
		std::string extension = findExtension(filename);
		int readerIndex = translateReaderExtension(extension);
		if (readerIndex < 0)
		{
			std::string message;
			message.append("Error: Could not deduce mesh reader type due to unknown file name extension '").append(extension).append("'");
			throw std::runtime_error(message);
		}

		return SWITCH_RETURN<ReaderAutoPtr, readersCount, IOReaderSelector, IOReaderDefault>::EXEC(readerIndex, filename);
	}

	static WriterAutoPtr getWriter(const char *filename)
	{
		std::string extension = findExtension(filename);
		int writerIndex = translateWriterExtension(extension);
		if (writerIndex < 0)
		{
			std::string message;
			message.append("Error: Could not deduce mesh writer type due to unknown file name extension '").append(extension).append("'");
			throw std::runtime_error(message);
		}

		return SWITCH_RETURN<WriterAutoPtr, writersCount, IOWriterSelector, IOWriterDefault>::EXEC(writerIndex, filename);
	}

private:
	static std::string findExtension(const std::string &filename)
	{
		size_t lastDot = filename.find_last_of('.');
		if (lastDot == std::string::npos)
			return "";

		return filename.substr(lastDot + 1);
	}

	static int translateReaderExtension(const std::string &extension)
	{
		for (int i = 0; i < readersCount; i++)
			if (extension.compare(readersExtensions[i]) == 0)
				return i;

		return -1;
	}

	static int translateWriterExtension(const std::string &extension)
	{
		for (int i = 0; i < writersCount; i++)
			if (extension.compare(writersExtensions[i]) == 0)
				return i;

		return -1;
	}
};


} // namespace implementation


#endif
