# Parametry
# dbg=(0, 1)            - 1...debug (default), 0...production
# o=(O2, O3, fast, ...) - optimalizace programu v production verzi (dbg=0), závisí na použitém kompilátoru
# omp=(0, 1)            - vypnutí/zapnutí překládání s OpenMP
# std=gccdefault        - vypnutí překládaní s parametrem -std=c++0x


EXECUTABLE := meshtest

TESTSOURCES := main.cpp

TESTHEADERS := mesh_info.h \
               test_configs.h


include ../src/files.mk


dbg ?= 1
o   ?= O3
omp ?= 1
std ?= c++0x


CXX  := g++
LINK := g++

CXXFLAGS += \
	-Wall \
	-Wpointer-arith \
	-Wreturn-type \
	-Wno-unused-function \
	-Wswitch \
	-Wformat \
	-Wchar-subscripts \
	-Wparentheses \
	-Wmultichar \
	-Wtrigraphs \
	-Wcast-align \


ROOTDIR       ?= ..
SRCDIR        ?= $(ROOTDIR)/src
BINDIR        ?= $(ROOTDIR)/bin
DEFOBJDIR     ?= $(ROOTDIR)/obj
TESTSRCDIR    ?= .
DEFTESTOBJDIR ?= $(TESTSRCDIR)/obj


INCLUDES += -I$(SRCDIR) -I$(TESTSRCDIR)
LIB      +=


CXXFLAGS += $(INCLUDES) -DUNIX

ifeq ($(dbg), 1)
	CXXFLAGS += -g -D_DEBUG

	OBJDIR     ?= $(DEFOBJDIR)/debug
	TESTOBJDIR ?= $(DEFTESTOBJDIR)/debug
else
	CXXFLAGS += -fno-strict-aliasing -DNDEBUG -$(o)

	OBJDIR     ?= $(DEFOBJDIR)
	TESTOBJDIR ?= $(DEFTESTOBJDIR)
endif


ifeq ($(omp), 1)
	CXXFLAGS += -fopenmp
	LIB      += -lgomp
endif


ifneq ($(std), gccdefault)
	CXXFLAGS += -std=$(std)
endif


OBJS += $(addprefix $(OBJDIR)/,$(SOURCES:.cpp=.o))
OBJS += $(addprefix $(TESTOBJDIR)/,$(TESTSOURCES:.cpp=.o))

DEPS += $(addprefix $(SRCDIR)/,$(HEADERS))
DEPS += $(addprefix $(TESTSRCDIR)/,$(TESTHEADERS))


TARGET := $(BINDIR)/$(EXECUTABLE)


.SUFFIXES : .cpp .o

# Rules #############################################################
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) -o $@ -c $<

$(TESTOBJDIR)/%.o: $(TESTSRCDIR)/%.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) -o $@ -c $<

$(EXECUTABLE): makedirs $(OBJS)
	$(LINK) -o $(TARGET) $(OBJS) $(LIB)


makedirs:
	@mkdir -p $(BINDIR)
	@mkdir -p $(OBJDIR)
	@mkdir -p $(TESTOBJDIR)

clean:
	@rm -f  $(TARGET)
	@rm -rf $(DEFOBJDIR)
	@rm -rf $(DEFTESTOBJDIR)
