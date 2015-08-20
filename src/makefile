CXX = g++
CXXFLAGS = -O3 -I. -I/usr/include/opencv2 -std=c++11
DEPS = functions.h segengine.h stdafx.h structures.h targetver.h utils.h
LIBS = -L/usr/lib -std=c++11 -lopencv_core -lopencv_highgui -lopencv_photo

%.o: %.cpp $(DEPS)
	$(CXX) -c -o $@ $< $(CXXFLAGS)

spixel: spixel.o segengine.o stdafx.o structures.o utils.o functions.o
	$(CXX) segengine.o utils.o functions.o spixel.o structures.o -o spixel $(CXXFLAGS) $(LIBS)
	
clean:
	rm -rf *.o
	rm spixel
	
install:
	install -D spixel /usr/local/bin/spixel
