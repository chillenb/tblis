if BUILD_BLIS
CONFIG_NAME = $(shell cat download/blis/config.mk | grep '^CONFIG_NAME' | awk '{print $$3}')
else
CONFIG_NAME = $(shell cat $(BLIS_PREFIX)/share/blis/config.mk | grep '^CONFIG_NAME' | awk '{print $$3}')
endif
SOURCE_SUFFIXES = .c .h .cxx .cpp .hpp .cc .hh .C
BLIS_H = $(BLIS_PREFIX)/include/blis/blis.h
BLIS_ARCHIVE = download/blis/lib/$(CONFIG_NAME)/libblis.a
PLUGIN_ARCHIVE = tblis/plugin/lib/$(CONFIG_NAME)/libblis_tblis.a
PLUGIN_SOURCE = $(foreach suf,$(SOURCE_SUFFIXES),$(shell find $(srcdir)/tblis/plugin -name \*$(suf)))

lib_LTLIBRARIES = lib/libtblis.la
EXTRA_lib_libtblis_la_DEPENDENCIES = $(PLUGIN_ARCHIVE)
lib_libtblis_la_SOURCES = \
    \
    tblis/frame/0/add.cxx \
    tblis/frame/0/mult.cxx \
    tblis/frame/0/reduce.cxx \
    \
    tblis/frame/1m/packm/packm_blk_bsmtc.cxx \
    tblis/frame/1m/packm/packm_blk_dpd.cxx \
    \
    tblis/frame/1t/dense/add.cxx \
    tblis/frame/1t/dense/dot.cxx \
    tblis/frame/1t/dense/reduce.cxx \
    tblis/frame/1t/dense/scale.cxx \
    tblis/frame/1t/dense/set.cxx \
    tblis/frame/1t/dense/shift.cxx \
    \
    tblis/frame/1t/dpd/add.cxx \
    tblis/frame/1t/dpd/dot.cxx \
    tblis/frame/1t/dpd/reduce.cxx \
    tblis/frame/1t/dpd/scale.cxx \
    tblis/frame/1t/dpd/set.cxx \
    tblis/frame/1t/dpd/shift.cxx \
    \
    tblis/frame/1t/indexed/add.cxx \
    tblis/frame/1t/indexed/dot.cxx \
    tblis/frame/1t/indexed/reduce.cxx \
    tblis/frame/1t/indexed/scale.cxx \
    tblis/frame/1t/indexed/set.cxx \
    tblis/frame/1t/indexed/shift.cxx \
    \
    tblis/frame/1t/indexed_dpd/add.cxx \
    tblis/frame/1t/indexed_dpd/dot.cxx \
    tblis/frame/1t/indexed_dpd/reduce.cxx \
    tblis/frame/1t/indexed_dpd/scale.cxx \
    tblis/frame/1t/indexed_dpd/set.cxx \
    tblis/frame/1t/indexed_dpd/shift.cxx \
    \
    tblis/frame/1t/add.cxx \
    tblis/frame/1t/dot.cxx \
    tblis/frame/1t/reduce.cxx \
    tblis/frame/1t/scale.cxx \
    tblis/frame/1t/set.cxx \
    tblis/frame/1t/shift.cxx \
    \
    tblis/frame/3m/gemm/gemm_ker_bsmtc.cxx \
    tblis/frame/3m/gemm/gemm_ker_dpd.cxx \
    \
    tblis/frame/3t/dense/mult.cxx \
    \
    tblis/frame/3t/dpd/mult.cxx \
    \
    tblis/frame/3t/indexed/mult.cxx \
    \
    tblis/frame/3t/indexed_dpd/mult.cxx \
    \
    tblis/frame/3t/mult.cxx \
    \
    tblis/frame/base/basic_types.cxx \
    tblis/frame/base/block_scatter.cxx \
    tblis/frame/base/dpd_block_scatter.cxx \
    tblis/frame/base/env.cxx \
    tblis/frame/base/tensor.cxx \
    tblis/frame/base/thread.cxx

include_HEADERS = tblis.h
pkginclude_HEADERS = tblis/tblis.h tblis/tblis_config.h

baseincludedir = $(pkgincludedir)/frame/base
baseinclude_HEADERS = \
    \
    tblis/frame/base/aligned_allocator.hpp \
    tblis/frame/base/alignment.hpp \
    tblis/frame/base/basic_types.h \
    tblis/frame/base/thread.h

frame1tincludedir = $(pkgincludedir)/frame/1t
frame1tinclude_HEADERS = \
    \
    tblis/frame/1t/add.h \
    tblis/frame/1t/dot.h \
    tblis/frame/1t/reduce.h \
    tblis/frame/1t/scale.h \
    tblis/frame/1t/set.h \
    tblis/frame/1t/shift.h

frame3tincludedir = $(pkgincludedir)/frame/3t
frame3tinclude_HEADERS = \
    \
    tblis/frame/3t/mult.h

marraydetailincludedir = $(includedir)/marray/detail
marraydetailinclude_HEADERS = \
    \
    tblis/external/marray/marray/detail/utility.hpp \
    tblis/external/marray/marray/detail/vector_avx.hpp \
    tblis/external/marray/marray/detail/vector_avx512.hpp \
    tblis/external/marray/marray/detail/vector_sse41.hpp \
    tblis/external/marray/marray/detail/vector.hpp

marrayfwdincludedir = $(includedir)/marray/fwd
marrayfwdinclude_HEADERS = \
    \
    tblis/external/marray/marray/fwd/expression_fwd.hpp \
    tblis/external/marray/marray/fwd/marray_fwd.hpp

marraydpdincludedir = $(includedir)/marray/dpd
marraydpdinclude_HEADERS = \
    \
    tblis/external/marray/marray/dpd/dpd_range.hpp \
    tblis/external/marray/marray/dpd/dpd_marray_base.hpp \
    tblis/external/marray/marray/dpd/dpd_marray_view.hpp \
    tblis/external/marray/marray/dpd/dpd_marray.hpp

marrayindexedincludedir = $(includedir)/marray/indexed
marrayindexedinclude_HEADERS = \
    \
    tblis/external/marray/marray/indexed/indexed_marray_base.hpp \
    tblis/external/marray/marray/indexed/indexed_marray_view.hpp \
    tblis/external/marray/marray/indexed/indexed_marray.hpp

marrayindexeddpdincludedir = $(includedir)/marray/indexed_dpd
marrayindexeddpdinclude_HEADERS = \
    \
    tblis/external/marray/marray/indexed_dpd/indexed_dpd_marray_base.hpp \
    tblis/external/marray/marray/indexed_dpd/indexed_dpd_marray_view.hpp \
    tblis/external/marray/marray/indexed_dpd/indexed_dpd_marray.hpp

marrayincludedir = $(includedir)/marray
marrayinclude_HEADERS = \
    \
    tblis/external/marray/marray/array_1d.hpp \
    tblis/external/marray/marray/array_2d.hpp \
    tblis/external/marray/marray/expression.hpp \
    tblis/external/marray/marray/index_iterator.hpp \
    tblis/external/marray/marray/marray_base.hpp \
    tblis/external/marray/marray/marray_iterator.hpp \
    tblis/external/marray/marray/marray_slice.hpp \
    tblis/external/marray/marray/marray_view.hpp \
    tblis/external/marray/marray/marray.hpp \
    tblis/external/marray/marray/range.hpp \
    tblis/external/marray/marray/rotate.hpp \
    tblis/external/marray/marray/short_vector.hpp \
    tblis/external/marray/marray/types.hpp

stl_extincludedir = $(includedir)/stl_ext
stl_extinclude_HEADERS = \
    \
    tblis/external/stl_ext/stl_ext/algorithm.hpp \
    tblis/external/stl_ext/stl_ext/any.hpp \
    tblis/external/stl_ext/stl_ext/bounded_vector.hpp \
    tblis/external/stl_ext/stl_ext/complex.hpp \
    tblis/external/stl_ext/stl_ext/cosort.hpp \
    tblis/external/stl_ext/stl_ext/fill_iterator.hpp \
    tblis/external/stl_ext/stl_ext/global_ptr.hpp \
    tblis/external/stl_ext/stl_ext/iostream.hpp \
    tblis/external/stl_ext/stl_ext/ptr_list.hpp \
    tblis/external/stl_ext/stl_ext/ptr_vector.hpp \
    tblis/external/stl_ext/stl_ext/string.hpp \
    tblis/external/stl_ext/stl_ext/type_traits.hpp \
    tblis/external/stl_ext/stl_ext/vector.hpp \
    tblis/external/stl_ext/stl_ext/zip.hpp

noinst_LTLIBRARIES =
lib_libtblis_la_LIBADD = tblis/external/tci/lib/libtci.la $(PLUGIN_ARCHIVE) -lblis

DISTCLEANFILES = tblis/tblis_config.h

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

SUBDIRS = tblis/external/tci
ACLOCAL_AMFLAGS = -I m4
AM_CPPFLAGS = -I$(srcdir) \
              -I$(srcdir)/tblis \
              -I$(srcdir)/tblis/external \
              -I$(srcdir)/tblis/external/tci \
              -I$(srcdir)/tblis/external/stl_ext \
              -I$(srcdir)/tblis/external/marray \
              -I$(srcdir)/tblis/external/catch \
              -I$(builddir) \
              -I$(builddir)/tblis \
              -I$(builddir)/tblis/util \
              -I$(builddir)/tblis/external/tci/tci \
              -I$(BLIS_PREFIX)/include
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

tblis/plugin/Makefile: $(PLUGIN_MAKEFILE_DEPS)
	@mkdir -p tblis/plugin
	@cd tblis/plugin && \
		$(BLIS_PREFIX)/share/blis/configure-plugin \
			CXX="$(CXX)" CXXFLAGS="$(CXXFLAGS) $(INCLUDE_PATHS) -I$(abs_srcdir)/tblis -I.. \
				-I$(abs_builddir)/tblis/external/tci/tci -I$(abs_srcdir)/tblis/external/tci" \
				-f --build --path=$(abs_srcdir)/tblis/plugin tblis

$(PLUGIN_ARCHIVE): tblis/plugin/Makefile
	$(MAKE) -C tblis/plugin

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
	@if [ -d tblis/plugin ]; then \
	target=`echo $@ | sed s/-local//`; \
	$(MAKE) -C tblis/plugin $$target; \
	fi
