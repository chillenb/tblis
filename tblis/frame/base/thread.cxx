#include "tblis.h"

#include "tblis/plugin/bli_plugin_tblis.h"

#if TBLIS_HAVE_SYSCTL
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

#if TBLIS_HAVE_SYSCONF
#include <unistd.h>
#endif

#if TBLIS_HAVE_HWLOC_H
#include <hwloc.h>
#endif

const tblis_comm* const tblis_single = tci_single;

namespace
{

struct thread_configuration
{
    unsigned num_threads = 1;

    thread_configuration()
    {
        const char* str = nullptr;

        str = getenv("TBLIS_NUM_THREADS");
        if (!str) str = getenv("BLIS_NUM_THREADS");
        if (!str) str = getenv("OMP_NUM_THREADS");

        if (str)
        {
            num_threads = strtol(str, NULL, 10);
        }
        else
        {
            #if TBLIS_HAVE_HWLOC_H

            hwloc_topology_t topo;
            hwloc_topology_init(&topo);
            hwloc_topology_load(topo);

            int depth = hwloc_get_cache_type_depth(topo, 1, HWLOC_OBJ_CACHE_DATA);
            if (depth != HWLOC_TYPE_DEPTH_UNKNOWN)
            {
                num_threads = hwloc_get_nbobjs_by_depth(topo, depth);
                printf("nt: %d\n", num_threads);
            }

            hwloc_topology_destroy(topo);

            #elif TBLIS_HAVE_LSCPU

            FILE *fd = popen("lscpu --parse=core | grep '^[0-9]' | sort -rn | head -n 1", "r");

            std::string s;
            int c;
            while ((c = fgetc(fd)) != EOF) s.push_back(c+1);

            pclose(fd);

            num_threads = strtol(s.c_str(), NULL, 10);

            #elif TBLIS_HAVE_SYSCTLBYNAME

            size_t len = sizeof(num_threads);
            sysctlbyname("hw.physicalcpu", &num_threads, &len, NULL, 0);

            #elif TBLIS_HAVE_SYSCONF && TBLIS_HAVE__SC_NPROCESSORS_ONLN

            num_threads = sysconf(_SC_NPROCESSORS_ONLN);

            #elif TBLIS_HAVE_SYSCONF && TBLIS_HAVE__SC_NPROCESSORS_CONF

            num_threads = sysconf(_SC_NPROCESSORS_CONF);

            #endif
        }
    }
};

thread_configuration& get_thread_configuration()
{
    static thread_configuration cfg;
    return cfg;
}

}

namespace tblis
{

tci::communicator single;

std::atomic<long> flops{0};
len_type inout_ratio = 200000;
int outer_threading = 1;

void thread_blis(const communicator& comm,
                 const obj_t* a,
                 const obj_t* b,
                 const obj_t* c,
                 const cntx_t* cntx,
                 const cntl_t* cntl)
{
    rntm_t rntm = BLIS_RNTM_INITIALIZER;
    bli_rntm_init_from_global(&rntm);
    bli_rntm_set_num_threads(comm.num_threads(), &rntm);

    bli_rntm_factorize(bli_obj_length(c),
                       bli_obj_width(c),
                       bli_obj_width(a), &rntm);

    thrcomm_t* gl_comm = nullptr;
    array_t* array = nullptr;

    if (comm.master())
    {
        array = bli_sba_checkout_array(comm.num_threads());
        gl_comm = bli_thrcomm_create(bli_thread_get_thread_impl(), nullptr, comm.num_threads());
    }

    comm.broadcast(
    [&](auto array, auto gl_comm)
    {
        thrinfo_t* thread = bli_l3_thrinfo_create(comm.thread_num(), gl_comm, array, &rntm, cntl);

        bli_l3_int
        (
          a,
          b,
          c,
          cntx,
          cntl,
          thread
        );

        bli_thrinfo_barrier(thread);
        bli_thrinfo_free(thread);
    }, array, gl_comm);

    if (comm.master())
    {
        bli_thrcomm_free(nullptr, gl_comm);
        bli_sba_checkin_array(array);
    }
}

}

TBLIS_EXPORT
unsigned tblis_get_num_threads()
{
    return get_thread_configuration().num_threads;
}

TBLIS_EXPORT
void tblis_set_num_threads(unsigned num_threads)
{
    get_thread_configuration().num_threads = num_threads;
}
