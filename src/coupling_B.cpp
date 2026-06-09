// ---------------------------------------------------------------------
//  src/coupling_B.cpp -- Case B: service-topology coupling.
//
//  Every physics rank submits cooling requests to a designated ML server
//  rank through the shared AsyncMpiTransport.  Server placement is
//  controlled at runtime by --ml-server-nodes=N (CouplingConfig):
//
//      N == 0  (default): one ML server rank per shared-memory node,
//                         chosen as the last rank on that node.
//      N  > 0 : N whole ML server nodes.  Compact node indices are
//               split into N contiguous groups; the last node in each
//               group is removed from physics work and all of its
//               ranks become ML servers for that group.
//
//  The implementation lives entirely in detail::ServiceCoupling and
//  detail::AsyncMpiTransport; this translation unit is only the
//  factory entry point for the CASE=B build.
// ---------------------------------------------------------------------

#include "mlhpc/coupling.hpp"
#include "mlhpc/detail/coupling_service.hpp"

#include <mpi.h>

#include <memory>

namespace mlhpc
{

std::unique_ptr<Coupling>
make_coupling (         MPI_Comm                  world,
               const    CouplingConfig           &cfg)
{
  return std::unique_ptr<Coupling>
    (new detail::ServiceCoupling (world, cfg));
}

const char *
compiled_case_label () noexcept
{
  return "B";
}

}  // namespace mlhpc
