#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_PENDINGWRITERANGES_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_PENDINGWRITERANGES_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <memory>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace AgcDriver::Graphics {

class PendingWriteRanges {
public:
    using Range = std::pair<std::uint64_t, std::uint64_t>;
    using Ranges = std::vector<Range>;

    struct Source {
        const void* key = nullptr;
        std::span<const Range> ranges;
        std::span<const std::uint64_t> notes;
    };

    struct Published {
        std::shared_ptr<const Ranges> snapshot;
        bool rebuilt = false;
    };

    Published Publish(std::span<const Source> sources, const std::shared_ptr<const Ranges>& current) {
        auto previous = std::move(published);
        published.reset();
        if (previous == nullptr || previous != current || normal == nullptr) return Rebuild(sources);
        ++pass;
        Ranges added;
        Ranges wrappedAdded;
        bool wrappedChanged = false;
        for (const auto& source : sources) {
            auto& entry = tracked[source.key];
            if (entry.pass == pass) continue;
            entry.pass = pass;
            const auto known = entry.ranges.size();
            if (known > source.ranges.size()) return Rebuild(sources);
            if (known != 0 && (source.notes.size() < known || source.notes[known - 1] != entry.lastNote)) return Rebuild(sources);
            for (auto index = known; index < source.ranges.size(); ++index) {
                const auto& range = source.ranges[index];
                entry.ranges.push_back(range);
                if (range.second > range.first) added.push_back(range);
                else if (range.second < range.first) wrappedAdded.push_back(range);
            }
            if (source.ranges.size() > known) entry.lastNote = source.notes.size() >= source.ranges.size() ? source.notes[source.ranges.size() - 1] : 0;
        }
        Ranges removed;
        for (auto it = tracked.begin(); it != tracked.end();) {
            if (it->second.pass == pass) {
                ++it;
                continue;
            }
            for (const auto& range : it->second.ranges) {
                if (range.second < range.first) {
                    wrapped.erase(wrapped.find(range));
                    wrappedChanged = true;
                }
                if (range.second <= range.first) continue;
                AddEdges(range, -1);
                removed.push_back(range);
            }
            it = tracked.erase(it);
        }
        for (const auto& range : wrappedAdded) wrapped.insert(range);
        wrappedChanged = wrappedChanged || !wrappedAdded.empty();
        if (added.empty() && removed.empty() && !wrappedChanged) {
            published = std::move(previous);
            return {published, false};
        }
        if (!added.empty() || !removed.empty()) {
            const bool removing = !removed.empty();
            auto next = removing ? Remove(*normal, std::move(removed)) : Ranges();
            for (const auto& range : added) AddEdges(range, 1);
            if (!added.empty()) next = Merge(removing ? next : *normal, std::move(added));
            normal = std::make_shared<const Ranges>(std::move(next));
        }
        return Finish(false);
    }

    std::size_t TrackedSources() const { return tracked.size(); }
    std::size_t Edges() const { return edges.size(); }

private:
    struct Tracked {
        Ranges ranges;
        std::uint64_t lastNote = 0;
        std::uint64_t pass = 0;
    };

    void AddEdges(const Range& range, std::int64_t count) {
        const auto add = [&](std::uint64_t at, std::int64_t delta) {
            const auto [it, inserted] = edges.try_emplace(at, 0);
            it->second += delta;
            if (it->second == 0) edges.erase(it);
        };
        add(range.first, count);
        add(range.second, -count);
    }

    void Sweep(std::map<std::uint64_t, std::int64_t>::const_iterator from, std::map<std::uint64_t, std::int64_t>::const_iterator to, Ranges& out) const {
        std::int64_t count = 0;
        std::uint64_t start = 0;
        for (auto it = from; it != to; ++it) {
            const auto before = count;
            count += it->second;
            if (before == 0 && count > 0) start = it->first;
            else if (before > 0 && count == 0) out.emplace_back(start, it->first);
        }
    }

    Published Finish(bool rebuilt) {
        published = wrapped.empty() ? normal : std::make_shared<const Ranges>(Merge(*normal, Ranges(wrapped.begin(), wrapped.end())));
        return {published, rebuilt};
    }

    Published Rebuild(std::span<const Source> sources) {
        tracked.clear();
        edges.clear();
        wrapped.clear();
        normal.reset();
        ++pass;
        for (const auto& source : sources) {
            auto& entry = tracked[source.key];
            if (entry.pass == pass) continue;
            entry.pass = pass;
            entry.ranges.assign(source.ranges.begin(), source.ranges.end());
            entry.lastNote = !source.ranges.empty() && source.notes.size() >= source.ranges.size() ? source.notes[source.ranges.size() - 1] : 0;
            for (const auto& range : source.ranges) {
                if (range.second > range.first) AddEdges(range, 1);
                else if (range.second < range.first) wrapped.insert(range);
            }
        }
        auto next = std::make_shared<Ranges>();
        Sweep(edges.cbegin(), edges.cend(), *next);
        normal = std::move(next);
        return Finish(true);
    }

    Ranges Remove(const Ranges& previous, Ranges removed) const {
        std::sort(removed.begin(), removed.end());
        Ranges out;
        out.reserve(previous.size() + removed.size());
        auto next = removed.cbegin();
        for (const auto& range : previous) {
            while (next != removed.cend() && next->second <= range.first) ++next;
            if (next == removed.cend() || next->first >= range.second) {
                out.push_back(range);
                continue;
            }
            Sweep(edges.lower_bound(range.first), edges.upper_bound(range.second), out);
        }
        return out;
    }

    static Ranges Merge(const Ranges& base, Ranges added) {
        std::sort(added.begin(), added.end());
        Ranges out;
        out.reserve(base.size() + added.size());
        auto left = base.cbegin();
        auto right = added.cbegin();
        while (left != base.cend() || right != added.cend()) {
            const bool takeLeft = right == added.cend() || (left != base.cend() && *left < *right);
            const auto& range = takeLeft ? *left++ : *right++;
            if (!out.empty() && range.first <= out.back().second) out.back().second = std::max(out.back().second, range.second);
            else out.push_back(range);
        }
        return out;
    }

    std::unordered_map<const void*, Tracked> tracked;
    std::map<std::uint64_t, std::int64_t> edges;
    std::multiset<Range> wrapped;
    std::shared_ptr<const Ranges> normal;
    std::shared_ptr<const Ranges> published;
    std::uint64_t pass = 0;
};

}

#endif
