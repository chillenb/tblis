#include "tensor.hpp"
#include "src/plugin/bli_plugin_tblis.h"

namespace tblis
{
namespace internal
{

void initialize_once()
{
    static auto initialized = []
    {
        register_plugin();
        return true;
    }();
}

}
}
