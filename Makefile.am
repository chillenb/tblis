if BUILD_BLIS
CONFIG_NAME = $(shell cat download/blis/config.mk | grep '^CONFIG_NAME' | awk '{print $$3}')
else
CONFIG_NAME = $(shell cat $(BLIS_PREFIX)/share/blis/config.mk | grep '^CONFIG_NAME' | awk '{print $$3}')
endif
SOURCE_SUFFIXES = .c .h .cxx .cpp .hpp .cc .hh .C
BLIS_H = $(BLIS_PREFIX)/include/blis/blis.h
BLIS_ARCHIVE = download/blis/lib/$(CONFIG_NAME)/libblis.a
PLUGIN_ARCHIVE = src/plugin/lib/$(CONFIG_NAME)/libblis_tblis.a
PLUGIN_SOURCE = $(foreach suf,$(SOURCE_SUFFIXES),$(shell find $(srcdir)/src/plugin -name \*$(suf)))

lib_LTLIBRARIES = lib/libtblis.la
EXTRA_lib_libtblis_la_DEPENDENCIES = $(PLUGIN_ARCHIVE)
lib_libtblis_la_SOURCES = \
    \
    src/frame/0/add.cxx \
    src/frame/0/mult.cxx \
    src/frame/0/reduce.cxx \
    \
    src/frame/1m/packm/packm_blk_bsmtc.cxx \
    src/frame/1m/packm/packm_blk_dpd.cxx \
    \
    src/frame/1t/dense/add.cxx \
    src/frame/1t/dense/dot.cxx \
    src/frame/1t/dense/reduce.cxx \
    src/frame/1t/dense/scale.cxx \
    src/frame/1t/dense/set.cxx \
    src/frame/1t/dense/shift.cxx \
    \
    src/frame/1t/dpd/add.cxx \
    src/frame/1t/dpd/dot.cxx \
    src/frame/1t/dpd/reduce.cxx \
    src/frame/1t/dpd/scale.cxx \
    src/frame/1t/dpd/set.cxx \
    src/frame/1t/dpd/shift.cxx \
    \
    src/frame/1t/indexed/add.cxx \
    src/frame/1t/indexed/dot.cxx \
    src/frame/1t/indexed/reduce.cxx \
    src/frame/1t/indexed/scale.cxx \
    src/frame/1t/indexed/set.cxx \
    src/frame/1t/indexed/shift.cxx \
    \
    src/frame/1t/indexed_dpd/add.cxx \
    src/frame/1t/indexed_dpd/dot.cxx \
    src/frame/1t/indexed_dpd/reduce.cxx \
    src/frame/1t/indexed_dpd/scale.cxx \
    src/frame/1t/indexed_dpd/set.cxx \
    src/frame/1t/indexed_dpd/shift.cxx \
    \
    src/frame/1t/add.cxx \
    src/frame/1t/dot.cxx \
    src/frame/1t/reduce.cxx \
    src/frame/1t/scale.cxx \
    src/frame/1t/set.cxx \
    src/frame/1t/shift.cxx \
    \
    src/frame/3m/gemm/gemm_ker_bsmtc.cxx \
    src/frame/3m/gemm/gemm_ker_dpd.cxx \
    \
    src/frame/3t/dense/mult.cxx \
    \
    src/frame/3t/dpd/mult.cxx \
    \
    src/frame/3t/indexed/mult.cxx \
    \
    src/frame/3t/indexed_dpd/mult.cxx \
    \
    src/frame/3t/mult.cxx \
    \
    src/frame/base/basic_types.cxx \
    src/frame/base/block_scatter.cxx \
    src/frame/base/dpd_block_scatter.cxx \
    src/frame/base/env.cxx \
    src/frame/base/tensor.cxx \
    src/frame/base/thread.cxx

pkginclude_HEADERS = src/tblis.h src/tblis_config.h

baseincludedir = $(pkgincludedir)/frame/base
baseinclude_HEADERS = \
    \
    src/frame/base/aligned_allocator.hpp \
    src/frame/base/alignment.hpp \
    src/frame/base/basic_types.h \
    src/frame/base/thread.h

frame1tincludedir = $(pkgincludedir)/frame/1t
frame1tinclude_HEADERS = \
    \
    src/frame/1t/add.h \
    src/frame/1t/dot.h \
    src/frame/1t/reduce.h \
    src/frame/1t/scale.h \
    src/frame/1t/set.h \
    src/frame/1t/shift.h

frame3tincludedir = $(pkgincludedir)/frame/3t
frame3tinclude_HEADERS = \
    \
    src/frame/3t/mult.h

marrayincludedir = $(pkgincludedir)/external/marray/marray
marrayinclude_HEADERS = \
    \
    src/external/marray/marray/detail/utility.hpp \
    src/external/marray/marray/detail/vector_avx.hpp \
    src/external/marray/marray/detail/vector_avx512.hpp \
    src/external/marray/marray/detail/vector_sse41.hpp \
    src/external/marray/marray/detail/vector.hpp \
    src/external/marray/marray/dpd/dpd_range.hpp \
    src/external/marray/marray/dpd/dpd_marray_base.hpp \
    src/external/marray/marray/dpd/dpd_marray_view.hpp \
    src/external/marray/marray/dpd/dpd_marray.hpp \
    src/external/marray/marray/fwd/expression_fwd.hpp \
    src/external/marray/marray/fwd/marray_fwd.hpp \
    src/external/marray/marray/indexed/indexed_marray_base.hpp \
    src/external/marray/marray/indexed/indexed_marray_view.hpp \
    src/external/marray/marray/indexed/indexed_marray.hpp \
    src/external/marray/marray/indexed_dpd/indexed_dpd_marray_base.hpp \
    src/external/marray/marray/indexed_dpd/indexed_dpd_marray_view.hpp \
    src/external/marray/marray/indexed_dpd/indexed_dpd_marray.hpp \
    src/external/marray/marray/array_1d.hpp \
    src/external/marray/marray/array_2d.hpp \
    src/external/marray/marray/expression.hpp \
    src/external/marray/marray/index_iterator.hpp \
    src/external/marray/marray/marray_base.hpp \
    src/external/marray/marray/marray_iterator.hpp \
    src/external/marray/marray/marray_slice.hpp \
    src/external/marray/marray/marray_view.hpp \
    src/external/marray/marray/marray.hpp \
    src/external/marray/marray/range.hpp \
    src/external/marray/marray/rotate.hpp \
    src/external/marray/marray/short_vector.hpp \
    src/external/marray/marray/types.hpp

stl_extincludedir = $(pkgincludedir)/external/stl_ext/stl_ext
stl_extinclude_HEADERS = \
    \
    src/external/stl_ext/stl_ext/algorithm.hpp \
    src/external/stl_ext/stl_ext/any.hpp \
    src/external/stl_ext/stl_ext/bounded_vector.hpp \
    src/external/stl_ext/stl_ext/complex.hpp \
    src/external/stl_ext/stl_ext/cosort.hpp \
    src/external/stl_ext/stl_ext/fill_iterator.hpp \
    src/external/stl_ext/stl_ext/global_ptr.hpp \
    src/external/stl_ext/stl_ext/iostream.hpp \
    src/external/stl_ext/stl_ext/ptr_list.hpp \
    src/external/stl_ext/stl_ext/ptr_vector.hpp \
    src/external/stl_ext/stl_ext/string.hpp \
    src/external/stl_ext/stl_ext/type_traits.hpp \
    src/external/stl_ext/stl_ext/vector.hpp \
    src/external/stl_ext/stl_ext/zip.hpp

noinst_LTLIBRARIES =
lib_libtblis_la_LIBADD = src/external/tci/lib/libtci.la $(PLUGIN_ARCHIVE) -lblis

DISTCLEANFILES = src/tblis_config.h

if IS_CLANG
CXXFLAGS += -Wno-pass-failed
endif

noinst_PROGRAMS = bin/test

EXTRA_bin_test_DEPENDENCIES = $(PLUGIN_ARCHIVE)
bin_test_SOURCES = test/test.cxx \
                   test/random.cxx \
                   \
                   test/1t/dot.cxx \
                   test/1t/reduce.cxx \
                   test/1t/scale.cxx \
                   test/1t/replicate.cxx \
                   test/1t/trace.cxx \
                   test/1t/transpose.cxx \
                   \
                   test/3m/gemm.cxx \
                   test/3m/gemv.cxx \
                   test/3m/ger.cxx \
                   \
                   test/3t/contract.cxx \
                   test/3t/mult.cxx \
                   test/3t/outer_prod.cxx \
                   test/3t/weight.cxx
#bin_skx_bench_SOURCES = test/skx_bench.cxx
#bin_bench_SOURCES = test/bench.cxx
#bin_batched_bench_SOURCES = test/batched_bench.cxx
#bin_dpd_bench_SOURCES = test/dpd_bench.cxx

VPATH += $(srcdir)

SUBDIRS = src/external/tci
ACLOCAL_AMFLAGS = -I m4
AM_CPPFLAGS = -I$(srcdir) -I$(srcdir)/src -I. -Isrc -Isrc/util -I$(srcdir)/src/external/tci -Isrc/external/tci/tci -I$(BLIS_PREFIX)/include
AM_LDFLAGS = -pthread -L$(BLIS_PREFIX)/lib
bin_test_LDADD = lib/libtblis.la

PLUGIN_MAKEFILE_DEPS = $(PLUGIN_SOURCE) Makefile
if BUILD_BLIS
PLUGIN_MAKEFILE_DEPS += $(BLIS_ARCHIVE)
endif

if BUILD_BLIS
BLIS_DEPS := $(shell find download/blis -name '*.c' -o -name '*.h')
endif

$(BLIS_ARCHIVE): $(BLIS_DEPS)
	@mkdir -p download/blis/installed
	$(MAKE) -C download/blis install

src/plugin/Makefile: $(PLUGIN_MAKEFILE_DEPS)
	@mkdir -p src/plugin
	@cd src/plugin && \
		$(BLIS_PREFIX)/share/blis/configure-plugin \
			CXX="$(CXX)" CXXFLAGS="$(CXXFLAGS) $(INCLUDE_PATHS) -I$(abs_srcdir)/src -I.. \
				-I$(abs_builddir)/src/external/tci/tci -I$(abs_srcdir)/src/external/tci" \
				-f --build --path=$(abs_srcdir)/src/plugin tblis

$(PLUGIN_ARCHIVE): src/plugin/Makefile
	$(MAKE) -C src/plugin

if BUILD_BLIS
$(BLIS_H): $(BLIS_ARCHIVE)
EXTRA_lib_libtblis_la_DEPENDENCIES += $(BLIS_ARCHIVE)
EXTRA_bin_test_DEPENDENCIES += $(BLIS_ARCHIVE)
endif

$(lib_libtblis_la_OBJECTS) $(bin_test_OBJECTS): $(BLIS_H)

clean-local:
if BUILD_BLIS
	@target=`echo $@ | sed s/-local//`; \
	$(MAKE) -C download/blis $$target
endif
	@if [ -d src/plugin ]; then \
	target=`echo $@ | sed s/-local//`; \
	$(MAKE) -C src/plugin $$target; \
	fi
