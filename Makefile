EXECUTABLE := test1
CU_FILES   := test1.cu
CU_DEPS    :=
# CC_FILES   := main.cpp


all: $(EXECUTABLE)

###########################################################

OBJDIR=objs
CXX=g++ -m64
CXXFLAGS=-O3 -Wall -g
LDFLAGS=-L/usr/local/cuda-11.7/lib64/ -lcudart
NVCC=nvcc
NVCCFLAGS=-O3 -m64 --gpu-architecture compute_61 -ccbin /usr/bin/gcc -G


OBJS=$(OBJDIR)/main.o  $(OBJDIR)/scan.o

.PHONY: dirs clean

all: $(EXECUTABLE)
                                    
default: $(EXECUTABLE)

dirs:
	mkdir -p $(OBJDIR)/

clean:
	rm -rf $(OBJDIR) *.ppm *~ $(EXECUTABLE)

check_scan: default
				./checker.pl scan

check_find_repeats: default
				./checker.pl find_repeats

$(EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)



$(OBJDIR)/%.o: %.cpp
	$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: %.cu
		(NVCC) $< $(NVCCFLAGS) -c -o $@

