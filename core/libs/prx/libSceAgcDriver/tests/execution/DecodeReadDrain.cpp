#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

using AgcDriver::DriverDetail::DrawPipeline;

void check(bool condition, const std::string& what) {
    if (!condition) throw std::runtime_error("decode read drain: " + what);
}

alignas(64) std::array<std::uint32_t, 4> AttributeTable{0x1fu, 0u, 0u, 0u};
alignas(64) std::array<std::uint32_t, 4> Elsewhere{};

std::uint64_t Address(const void* data) {
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(data));
}

void WaitEmpty() {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (DrawPipeline::QueuedItems() != 0) {
        check(std::chrono::steady_clock::now() < deadline, "the pipeline never emptied");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

}

int main() {
    try {
        DrawPipeline::Active() = true;
        auto& pipeline = DrawPipeline::Queue0();
        const auto table = Address(AttributeTable.data());

        // A queued item that stores a new attribute word when it commits, held back for a while: a
        // vertex-table read of that word (Driver::draw's decodeVertexInfo) must drain it first.
        std::atomic<bool> release{false};
        pipeline.Enqueue([&release] {
            while (!release.load(std::memory_order_acquire)) std::this_thread::sleep_for(std::chrono::milliseconds(1));
            AttributeTable[0] = 0x00000021u;
        }, {{table, table + sizeof(AttributeTable)}});
        std::thread releaser([&release] {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            release.store(true, std::memory_order_release);
        });
        const bool drained = DrawPipeline::DrainBeforeRead(table, 4);
        releaser.join();
        check(drained, "a read of a range a queued item writes did not drain");
        check(AttributeTable[0] == 0x00000021u, "the read would see the attribute word before the queued store");
        check(DrawPipeline::QueuedItems() == 0, "the drain left the store queued");

        // A read elsewhere keeps the queued item in place.
        std::atomic<bool> second{false};
        pipeline.Enqueue([&second] {
            while (!second.load(std::memory_order_acquire)) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }, {{Address(Elsewhere.data()), Address(Elsewhere.data()) + sizeof(Elsewhere)}});
        check(!DrawPipeline::DrainBeforeRead(table, 4), "a read that overlaps nothing queued drained the pipeline");
        check(DrawPipeline::QueuedItems() == 1, "the unrelated queued item is gone");
        second.store(true, std::memory_order_release);
        WaitEmpty();

        // Off the queue-0 worker (DrawPipeline::Active false) nothing is drained.
        DrawPipeline::Active() = false;
        check(!DrawPipeline::DrainBeforeRead(table, 4), "a read off the queue-0 worker drained");
        std::puts("decode read drain tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
