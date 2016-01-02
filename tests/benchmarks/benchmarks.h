#pragma once

#include <iostream>
#include <iomanip>
#include <map>
#include <vector>

#include <core/tnlTimerRT.h>
#include <core/tnlString.h>

namespace tnl
{
namespace benchmarks
{

const double oneGB = 1024.0 * 1024.0 * 1024.0;

template< typename ComputeFunction,
          typename ResetFunction >
double
timeFunction( ComputeFunction compute,
              ResetFunction reset,
              const int & loops )
{
    // the timer is constructed zero-initialized and stopped
    tnlTimerRT timer;

    reset();
    for(int i = 0; i < loops; ++i) {
        // Explicit synchronization of the CUDA device
        // TODO: not necessary for host computations
#ifdef HAVE_CUDA
        cudaDeviceSynchronize();
#endif
        timer.start();
        compute();
#ifdef HAVE_CUDA
        cudaDeviceSynchronize();
#endif
        timer.stop();

        reset();
    }

    return timer.getTime();
}


struct InternalError {};


class Logging
{
public:
    using MetadataElement = std::pair< const char*, tnlString >;
    using MetadataMap = std::map< const char*, tnlString >;
    using MetadataColumns = std::vector<MetadataElement>;

    using HeaderElements = std::initializer_list< tnlString >;
    using RowElements = std::initializer_list< double >;

    Logging( bool verbose = true )
        : verbose(verbose)
    { }

    void
    writeTitle( const tnlString & title )
    {
        if( verbose )
            std::cout << std::endl << "== " << title << " ==" << std::endl << std::endl;
        log << ": title = " << title << std::endl;
    }

    void
    writeMetadata( const MetadataMap & metadata )
    {
        if( verbose )
            std::cout << "properties:" << std::endl;

        for( auto & it : metadata ) {
            if( verbose )
                std::cout << "   " << it.first << " = " << it.second << std::endl;
            log << ": " << it.first << " = " << it.second << std::endl;
        }
        if( verbose )
            std::cout << std::endl;
    }

    void
    writeTableHeader( const tnlString & spanningElement,
                      const HeaderElements & subElements )
    {
        using namespace std;

        if( verbose && header_changed ) {
            for( auto & it : metadataColumns ) {
                cout << setw( 20 ) << it.first;
            }

            // spanning element is printed as usual column to stdout,
            // but is excluded from header
            cout << setw( 15 ) << "";

            for( auto & it : subElements ) {
                cout << setw( 15 ) << it;
            }
            cout << endl;

            header_changed = false;
        }

        // initial indent string
        header_indent = "!";
        log << endl;
        for( auto & it : metadataColumns ) {
            log << header_indent << " " << it.first << endl;
        }

        // dump stacked spanning columns
        if( horizontalGroups.size() > 0 )
            while( horizontalGroups.back().second <= 0 ) {
                horizontalGroups.pop_back();
                header_indent.pop_back();
            }
        for( int i = 0; i < horizontalGroups.size(); i++ ) {
            if( horizontalGroups[ i ].second > 0 ) {
                log << header_indent << " " << horizontalGroups[ i ].first << endl;
                header_indent += "!";
            }
        }

        log << header_indent << " " << spanningElement << endl;
        for( auto & it : subElements ) {
            log << header_indent << "! " << it << endl;
        }

        if( horizontalGroups.size() > 0 ) {
            horizontalGroups.back().second--;
            header_indent.pop_back();
        }
    }

    void
    writeTableRow( const tnlString & spanningElement,
                   const RowElements & subElements )
    {
        using namespace std;

        if( verbose ) {
            for( auto & it : metadataColumns ) {
                cout << setw( 20 ) << it.second;
            }
            // spanning element is printed as usual column to stdout
            cout << setw( 15 ) << spanningElement;
            for( auto & it : subElements ) {
                cout << setw( 15 );
                if( it != 0.0 ) cout << it;
                else cout << "N/A";
            }
            cout << endl;
        }

        // only when changed (the header has been already adjusted)
        // print each element on separate line
        for( auto & it : metadataColumns ) {
            log << it.second << endl;
        }

        // benchmark data are indented
        const tnlString indent = "    ";
        for( auto & it : subElements ) {
            if( it != 0.0 ) log << indent << it << endl;
            else log << indent << "N/A" << endl;
        }
    }

    void
    closeTable()
    {
        header_indent = body_indent = "";
        header_changed = true;
    }

    bool save( std::ostream & logFile )
    {
        closeTable();
        logFile << log.str();
        if( logFile.good() ) {
            log.str() ="";
            return true;
        }
        return false;
    }

protected:

    // manual double -> tnlString conversion with fixed precision
    static tnlString
    _to_string( const double & num, const int & precision = 0, bool fixed = false )
    {
        std::stringstream str;
        if( fixed )
            str << std::fixed;
        if( precision )
            str << std::setprecision( precision );
        str << num;
        return tnlString( str.str().data() );
    }

    std::stringstream log;
    std::string header_indent;
    std::string body_indent;

    bool verbose;
    MetadataColumns metadataColumns;
    bool header_changed = true;
    std::vector< std::pair< tnlString, int > > horizontalGroups;
};


class Benchmark
    : protected Logging
{
public:
    using Logging::MetadataElement;
    using Logging::MetadataMap;
    using Logging::MetadataColumns;

    Benchmark( const int & loops = 10,
               bool verbose = true )
        : Logging(verbose), loops(loops)
    { }

    // TODO: ensure that this is not called in the middle of the benchmark
    // (or just remove it completely?)
    void
    setLoops( const int & loops )
    {
        this->loops = loops;
    }

    // Marks the start of a new benchmark
    void
    newBenchmark( const tnlString & title )
    {
        closeTable();
        writeTitle( title );
    }

    // Marks the start of a new benchmark (with custom metadata)
    void
    newBenchmark( const tnlString & title,
                  MetadataMap metadata )
    {
        closeTable();
        writeTitle( title );
        // add loops to metadata
        metadata["loops"] = tnlString(loops);
        writeMetadata( metadata );
    }

    // Sets metadata columns -- values used for all subsequent rows until
    // the next call to this function.
    void
    setMetadataColumns( const MetadataColumns & metadata )
    {
        if( metadataColumns != metadata )
            header_changed = true;
        metadataColumns = metadata;
    }

    // TODO: maybe should be renamed to createVerticalGroup and ensured that vertical and horizontal groups are not used within the same "Benchmark"
    // Sets current operation -- operations expand the table vertically
    //  - baseTime should be reset to 0.0 for most operations, but sometimes
    //    it is useful to override it
    //  - Order of operations inside a "Benchmark" does not matter, rows can be
    //    easily sorted while converting to HTML.)
    void
    setOperation( const tnlString & operation,
                  const double & datasetSize = 0.0, // in GB
                  const double & baseTime = 0.0 )
    {
        if( metadataColumns.size() > 0 && tnlString(metadataColumns[ 0 ].first) == "operation" ) {
            metadataColumns[ 0 ].second = operation;
        }
        else {
            metadataColumns.insert( metadataColumns.begin(), {"operation", operation} );
        }
        setOperation( datasetSize, baseTime );
        header_changed = true;
    }

    void
    setOperation( const double & datasetSize = 0.0,
                  const double & baseTime = 0.0 )
    {
        this->datasetSize = datasetSize;
        this->baseTime = baseTime;
    }

    // Creates new horizontal groups inside a benchmark -- increases the number
    // of columns in the "Benchmark", implies column spanning.
    // (Useful e.g. for SpMV formats, different configurations etc.)
    void
    createHorizontalGroup( const tnlString & name,
                           const int & subcolumns )
    {
        if( horizontalGroups.size() == 0 ) {
            horizontalGroups.push_back( {name, subcolumns} );
        }
        else {
            auto & last = horizontalGroups.back();
            if( last.first != name && last.second > 0 ) {
                horizontalGroups.push_back( {name, subcolumns} );
            }
            else {
                last.first = name;
                last.second = subcolumns;
            }
        }
    }

    // Times a single ComputeFunction. Subsequent calls implicitly split
    // the current "horizontal group" into sub-columns identified by
    // "performer", which are further split into "bandwidth", "time" and
    // "speedup" columns.
    // TODO: allow custom columns bound to lambda functions (e.g. for Gflops calculation)
    // Also terminates the recursion of the following variadic template.
    template< typename ResetFunction,
              typename ComputeFunction >
    double
    time( ResetFunction reset,
          const tnlString & performer,
          ComputeFunction & compute )
    {
        const double time = timeFunction( compute, reset, loops );
        const double bandwidth = datasetSize / time;
        const double speedup = this->baseTime / time;
        if( this->baseTime == 0.0 )
            this->baseTime = time;

        writeTableHeader( performer, HeaderElements({"bandwidth", "time", "speedup"}) );
        writeTableRow( performer, RowElements({ bandwidth, time, speedup }) );

        return this->baseTime;
    }

    // Recursive template function to deal with multiple computations with the
    // same reset function.
    template< typename ResetFunction,
              typename ComputeFunction,
              typename... NextComputations >
    inline double
    time( ResetFunction reset,
          const tnlString & performer,
          ComputeFunction & compute,
          NextComputations & ... nextComputations )
    {
        time( reset, performer, compute );
        time( reset, nextComputations... );
        return this->baseTime;
    }

    using Logging::save;

protected:
    int loops;
    double datasetSize = 0.0;
    double baseTime = 0.0;
};

} // namespace benchmarks
} // namespace tnl
