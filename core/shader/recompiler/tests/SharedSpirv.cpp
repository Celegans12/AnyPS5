#include "Recompiler.hpp"
#include "ColdSpirvCodec.hpp"
#include "ColdSpirvCodecTestHooks.hpp"
#include <iostream>
#include <stdexcept>

namespace {

using namespace ShaderRecompiler;
namespace Testing = ColdSpirv::Testing;

void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template<class F>
void RequireFailure(F&& action) {
    bool failed = false;
    try { action(); }
    catch (const std::exception&) { failed = true; }
    Testing::ResetAllocationFailure();
    Require(failed, "injected allocation failure did not reach the storage operation");
    Require(Testing::ActiveAllocations() == 0 && Testing::ActiveBytes() == 0, "failed storage operation leaked codec allocations");
}

void AllocationFailurePreservesStorage() {
    const std::vector<std::uint32_t> expected(32u * 1024u, 0x07230203u);
    SharedSpirv raw(expected);
    Testing::FailAllocationAfter(0);
    Require(!raw.TryMakeCold(), "failed compression reported successful demotion");
    Testing::ResetAllocationFailure();
    Require(raw.ColdBytes() == 0 && raw.Words() == expected, "failed compression lost resident words");

    const auto compressed = ColdSpirv::Compress(expected);
    Require(compressed.has_value(), "allocation fixture did not compress");
    Testing::ResetAllocationFailure();
    Require(ColdSpirv::Decompress(*compressed, expected.size()) == expected, "allocation fixture did not decode");
    const auto checks = Testing::AllocationChecks();
    for (std::size_t failAt = 0; failAt < checks; ++failAt) {
        SharedSpirv cold(expected);
        Require(cold.TryMakeCold(), "hydration fixture did not become cold");
        const auto bytes = cold.ColdBytes();
        Testing::FailAllocationAfter(failAt);
        RequireFailure([&] { static_cast<void>(cold.Words()); });
        Require(cold.ColdBytes() == bytes && cold.InspectStorage().residentBytes == 0, "failed hydration discarded its compressed source");
        Require(cold.Words() == expected && cold.ColdBytes() == 0, "hydration could not retry after allocation failure");
    }

    SharedSpirv cold(expected);
    Require(cold.TryMakeCold(), "transient fixture did not become cold");
    const auto bytes = cold.ColdBytes();
    Testing::FailAllocationAfter(0);
    RequireFailure([&] { static_cast<void>(cold.ReadTransient()); });
    Require(cold.ColdBytes() == bytes && !cold.InspectStorage().exposed, "failed transient read changed cold storage");
    Require(*cold.ReadTransient() == expected && cold.ColdBytes() == bytes, "transient read could not retry without hydration");

    auto copy = cold;
    Testing::FailAllocationAfter(0);
    RequireFailure([&] { copy[0] = 1u; });
    Require(copy.ColdBytes() == bytes && cold.ColdBytes() == bytes, "failed cold copy-on-write discarded shared storage");
    copy[0] = 1u;
    Require(cold.ColdBytes() == bytes && copy.Words()[0] == 1u && *cold.ReadTransient() == expected, "copy-on-write retry changed its cold sibling");
}

}

int main() {
    try {
        AllocationFailurePreservesStorage();
        std::cout << "cold SPIR-V storage allocation tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        Testing::ResetAllocationFailure();
        std::cerr << error.what() << '\n';
        return 1;
    }
}
