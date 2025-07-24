all: shared static

shared: test_install.o
	$(CXX) -o shared.x $< $(shell pkg-config --libs TBLIS)

static: test_install.o
	$(CXX) -o static.x $< $(shell pkg-config --libs --static TBLIS)

test_install.o: test_install.cxx
	$(CXX) -c -o $@ $< -std=c++20 $(shell pkg-config --cflags TBLIS)

