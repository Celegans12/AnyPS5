#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_SRC_DRIVER_NEWPATH_DEVICEPAGETABLE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_SRC_DRIVER_NEWPATH_DEVICEPAGETABLE_HPP

#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/CommandAccess.hpp"
#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/CommandRecord.hpp"
#include "prx/libSceAgcDriver/Graphics/include/BdaPageTable.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <span>
#include <vector>

namespace AgcDriver::NewPath {

class DevicePageTable {
public:
    using Range = ShaderRecompiler::BdaAbi::Range;
    using ImportSource = std::function<Graphics::HostImportSnapshot(const Graphics::Context&)>;
    using VersionSource = std::function<std::uint64_t(VkDevice)>;
    using ResidentPieces = std::function<std::vector<Range>(std::uint64_t begin, std::uint64_t end)>;
    // The read-only image mirrors an address-based build binds in place (exe ranges no import can
    // serve), mapped at the build's own device address; versioned apart from the imports.
    using MirrorSource = std::function<Graphics::ImageMirrorTableSnapshot(const Graphics::Context&)>;

    struct Counts {
        std::uint64_t builds = 0;
        std::uint64_t patches = 0;
        std::uint64_t unchanged = 0;
        std::uint64_t ranges = 0;
        std::uint64_t coveredBytes = 0;
        std::uint64_t importVersion = 0;
        std::uint64_t mirrorVersion = 0;
        std::uint64_t mirrorRanges = 0;
        std::uint64_t mirrorBytes = 0;
        std::uint64_t released = 0;
        std::uint64_t outlived = 0;
        std::uint64_t records = 0;
        std::array<std::uint64_t, 4> coverHits{};
        std::uint64_t serialHits = 0;
        std::uint64_t compares = 0;
    };

    enum class Cover : std::uint8_t { Covered, Missing, OtherAddress, Narrower };

    struct Snapshot {
        std::shared_ptr<const Graphics::BdaPageTable> table;
        TableLists lists;
    };

    DevicePageTable(ImportSource imports, VersionSource version, MirrorSource mirrors = {}, VersionSource mirrorVersion = {});

    std::shared_ptr<const Graphics::BdaPageTable> Refresh(const Graphics::Context& context);
    [[nodiscard]] bool Stale(VkDevice device) const;
    std::shared_ptr<const Graphics::BdaPageTable> PatchPages(const Graphics::Context& context, std::span<const std::uint64_t> pages);
    void SetResidentPieces(ResidentPieces pieces);
    void Release(VkDevice device);

    [[nodiscard]] std::shared_ptr<const Graphics::BdaPageTable> Current() const;
    [[nodiscard]] Counts Totals() const;
    [[nodiscard]] std::vector<Range> Ranges() const;
    [[nodiscard]] std::uint64_t RootAddress() const;
    [[nodiscard]] TableLists Lists() const;
    [[nodiscard]] std::shared_ptr<Graphics::Buffer> TakeRecord(const Graphics::Context& context);
    void ReturnRecord(std::shared_ptr<Graphics::Buffer> record, VkDevice device);
    [[nodiscard]] std::size_t PooledRecords() const;

    [[nodiscard]] Cover Covers(const std::shared_ptr<const std::vector<Range>>& build, Snapshot* used = nullptr);
    [[nodiscard]] Cover Covers(std::uint64_t serial, std::span<const Range> build, Snapshot* used = nullptr);
    [[nodiscard]] Snapshot Use() const;
    [[nodiscard]] Cover Probe(std::span<const Range> build, Range* failed = nullptr) const;
    [[nodiscard]] static Cover Compare(std::span<const Range> table, std::span<const Range> build, const Range** failed = nullptr, std::uint64_t* failedAt = nullptr);
    // The imports (sorted, disjoint) with the mirrors added where no import maps: a mirror's bytes
    // that an import also maps are left to the import. Sorted and disjoint.
    [[nodiscard]] static std::vector<Range> WithMirrors(std::vector<Range> imports, std::span<const Range> mirrors);
    [[nodiscard]] static std::vector<Range> Overlay(std::span<const Range> base, std::span<const Range> resident, std::uint64_t begin, std::uint64_t end);

private:
    [[nodiscard]] std::vector<Range> piecesLocked(std::uint64_t begin, std::uint64_t end) const;

    [[nodiscard]] std::uint64_t mirrorVersionOf(VkDevice device) const;
    [[nodiscard]] Cover compareLocked(std::span<const Range> build);

    ImportSource imports;
    VersionSource version;
    MirrorSource mirrors;
    VersionSource mirrorVersion;
    std::vector<std::shared_ptr<const void>> mirrorHolds;
    std::uint64_t builtMirrorVersion = 0;
    mutable std::mutex mutex;
    ResidentPieces resident;
    std::vector<Range> importRanges;
    std::vector<Range> tableRanges;
    TableLists tableLists;
    std::vector<std::shared_ptr<Graphics::Buffer>> records;
    VkDevice recordDevice = VK_NULL_HANDLE;
    std::uint64_t recordsMade = 0;
    std::uint64_t builtVersion = 0;
    bool built = false;
    std::shared_ptr<const Graphics::BdaPageTable> table;
    VkDevice tableDevice = VK_NULL_HANDLE;
    std::vector<std::shared_ptr<const Graphics::BdaPageTable>> outlivedDevices;
    Counts counts;
    std::uint64_t generation = 0;
    struct CoverMemo {
        std::shared_ptr<const std::vector<Range>> build;
        std::uint64_t generation = 0;
        Cover cover = Cover::Missing;
    };
    std::array<CoverMemo, 16> coverMemo{};
    std::size_t coverNext = 0;
    std::unordered_map<std::uint64_t, Cover> serialMemo;
    std::uint64_t serialMemoGeneration = 0;
    std::size_t reportedFailures = 0;
};

DevicePageTable& DeviceTable();

}

#endif
