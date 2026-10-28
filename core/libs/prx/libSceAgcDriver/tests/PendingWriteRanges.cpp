#include "prx/libSceAgcDriver/Graphics/include/PendingWriteRanges.hpp"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <list>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using AgcDriver::Graphics::PendingWriteRanges;
using Range = PendingWriteRanges::Range;
using Ranges = PendingWriteRanges::Ranges;

struct Batch {
    Ranges writes;
    std::vector<std::uint64_t> notes;
};

struct Model {
    std::list<std::unique_ptr<Batch>> batches;
    std::uint64_t noteCount = 0;
    PendingWriteRanges set;
    std::shared_ptr<const Ranges> current;
    std::uint64_t rebuilds = 0;

    void Note(Batch& batch, std::uint64_t begin, std::uint64_t end) {
        batch.writes.emplace_back(begin, end);
        batch.notes.push_back(++noteCount);
    }

    Ranges Reference() const {
        Ranges all;
        for (const auto& batch : batches) all.insert(all.end(), batch->writes.begin(), batch->writes.end());
        std::erase_if(all, [](const Range& range) { return range.second == range.first; });
        std::sort(all.begin(), all.end());
        Ranges merged;
        for (const auto& range : all) {
            if (!merged.empty() && range.first <= merged.back().second) merged.back().second = std::max(merged.back().second, range.second);
            else merged.push_back(range);
        }
        return merged;
    }

    bool Publish() {
        std::vector<PendingWriteRanges::Source> sources;
        for (const auto& batch : batches) sources.push_back({batch.get(), batch->writes, batch->notes});
        auto published = set.Publish(sources, current);
        current = std::move(published.snapshot);
        if (published.rebuilt) ++rebuilds;
        if (current == nullptr) throw std::runtime_error("published no snapshot");
        return published.rebuilt;
    }

    void Check(const std::string& what) const {
        if (*current != Reference()) throw std::runtime_error("snapshot differs from the merged union of the live batches: " + what);
    }
};

void Require(bool condition, const std::string& what) {
    if (!condition) throw std::runtime_error(what);
}

void MergesAndRemovesWithoutRebuild() {
    Model model;
    model.batches.push_back(std::make_unique<Batch>());
    auto& first = *model.batches.back();
    model.Note(first, 0x1000, 0x2000);
    Require(model.Publish(), "the first publication rebuilds");
    model.Check("first range");
    model.Note(first, 0x3000, 0x4000);
    Require(!model.Publish(), "a recorded range merges without a rebuild");
    model.Check("second range");
    model.Note(first, 0x2000, 0x3000);
    Require(!model.Publish(), "a range joining two published ranges merges");
    model.Check("joined ranges");
    Require(model.current->size() == 1, "touching ranges are one published range");
    model.batches.push_back(std::make_unique<Batch>());
    auto& second = *model.batches.back();
    model.Note(second, 0x1800, 0x2800);
    model.Note(second, 0x9000, 0xa000);
    Require(!model.Publish(), "a second batch merges");
    model.Check("second batch");
    const auto before = model.current;
    Require(!model.Publish(), "an unchanged publication does not rebuild");
    Require(model.current == before, "an unchanged publication keeps the snapshot");
    model.batches.pop_front();
    Require(!model.Publish(), "a retired batch is removed without a rebuild");
    model.Check("first batch retired");
    Require(*model.current == Ranges{{0x1800, 0x2800}, {0x9000, 0xa000}}, "the retired batch's ranges leave only the second batch's");
    Require(*before == Ranges{{0x1000, 0x4000}, {0x9000, 0xa000}}, "a publication left an earlier snapshot unchanged");
    model.batches.pop_front();
    Require(!model.Publish(), "the last batch retires without a rebuild");
    Require(model.current->empty(), "no batch leaves an empty snapshot");
    Require(model.set.TrackedSources() == 0 && model.set.Edges() == 0, "retired batches leave no tracked state");
    Require(model.rebuilds == 1, "only the first publication rebuilt");
}

void CoveredNotesStayCounted() {
    Model model;
    model.batches.push_back(std::make_unique<Batch>());
    model.Note(*model.batches.back(), 0x1000, 0x2000);
    model.Publish();
    model.batches.push_back(std::make_unique<Batch>());
    model.Note(*model.batches.back(), 0x1400, 0x1800);
    model.batches.push_back(std::make_unique<Batch>());
    model.Note(*model.batches.back(), 0x1000, 0x2000);
    model.batches.pop_front();
    Require(!model.Publish(), "notes recorded without a publication merge at the next one");
    model.Check("a covered note outlives the batch that covered it");
    Require(*model.current == Ranges{{0x1000, 0x2000}}, "the covered notes keep their range published");
}

void ChangedSourcesRebuild() {
    Model model;
    model.batches.push_back(std::make_unique<Batch>());
    auto* batch = model.batches.back().get();
    model.Note(*batch, 0x1000, 0x2000);
    model.Note(*batch, 0x5000, 0x6000);
    model.Publish();
    batch->writes.clear();
    batch->notes.clear();
    model.Note(*batch, 0x7000, 0x8000);
    model.Note(*batch, 0x8000, 0x9000);
    Require(model.Publish(), "a source whose recorded ranges were replaced rebuilds");
    model.Check("replaced ranges");
    batch->writes.pop_back();
    batch->notes.pop_back();
    Require(model.Publish(), "a source that lost ranges rebuilds");
    model.Check("shrunk source");
    model.current = std::make_shared<const Ranges>(Ranges{{0, 1}});
    Require(model.Publish(), "a snapshot published by another owner rebuilds");
    model.Check("foreign snapshot");
    model.current = nullptr;
    Require(model.Publish(), "a cleared snapshot rebuilds");
    model.Check("cleared snapshot");
}

void EmptyAndWrappedRanges() {
    Model model;
    model.batches.push_back(std::make_unique<Batch>());
    model.Note(*model.batches.back(), 0x1000, 0x1000);
    model.Publish();
    Require(model.current->empty(), "an empty range publishes nothing");
    model.Note(*model.batches.back(), 0x2000, 0x1000);
    model.Note(*model.batches.back(), 0x3000, 0x3100);
    Require(!model.Publish(), "empty and wrapped ranges merge without a rebuild");
    model.Check("empty and wrapped ranges");
    Require(*model.current == Ranges{{0x2000, 0x1000}, {0x3000, 0x3100}}, "a wrapped range stays in the union as the merged reference keeps it");
    model.batches.push_back(std::make_unique<Batch>());
    model.Note(*model.batches.back(), 0x1800, 0x2800);
    model.Note(*model.batches.back(), UINT64_MAX - 3, 4);
    Require(!model.Publish(), "a range covering a wrapped range merges without a rebuild");
    model.Check("a wrapped range inside a range");
    model.batches.pop_back();
    Require(!model.Publish(), "a retired batch with a wrapped range is removed without a rebuild");
    model.Check("a retired wrapped range");
    model.batches.clear();
    model.Publish();
    Require(model.current->empty() && model.set.Edges() == 0, "retired empty ranges leave no edges");
}

void RandomSequencesMatchTheUnion() {
    std::mt19937_64 random(0x5eedu);
    Model model;
    std::vector<std::unique_ptr<Batch>> retired;
    for (int step = 0; step < 20000; ++step) {
        const auto action = random() % 10;
        if (action < 6 || model.batches.empty()) {
            if (model.batches.empty() || random() % 8 == 0) model.batches.push_back(std::make_unique<Batch>());
            auto it = model.batches.begin();
            std::advance(it, static_cast<long>(random() % model.batches.size()));
            const auto notes = 1 + random() % 3;
            for (std::uint64_t index = 0; index < notes; ++index) {
                const auto begin = (random() % 512) * 0x100;
                const auto end = random() % 32 == 0 ? (random() % 64) * 0x100 : begin + (random() % 16) * 0x100;
                model.Note(**it, begin, end);
            }
        } else if (action < 9) {
            auto it = model.batches.begin();
            std::advance(it, static_cast<long>(random() % std::min<std::size_t>(model.batches.size(), 2)));
            retired.push_back(std::move(*it));
            model.batches.erase(it);
        }
        if (random() % 4 != 0) {
            model.Publish();
            model.Check("random step " + std::to_string(step));
        }
    }
    Require(model.rebuilds == 1, "random appends and retirements never rebuilt after the first publication (" + std::to_string(model.rebuilds) + " rebuilds)");
}

}

int main() {
    try {
        MergesAndRemovesWithoutRebuild();
        CoveredNotesStayCounted();
        ChangedSourcesRebuild();
        EmptyAndWrappedRanges();
        RandomSequencesMatchTheUnion();
        std::puts("pending write ranges merge on record, leave on retirement and rebuild only on changed sources");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
