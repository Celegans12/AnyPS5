#include "prx/libSceAgcDriver/Execution/include/AsyncLog.hpp"
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void Check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

std::string ReadAll(std::FILE* file) {
    std::fflush(file);
    std::rewind(file);
    std::string text;
    char buffer[4096];
    for (std::size_t read = 0; (read = std::fread(buffer, 1, sizeof(buffer), file)) != 0;) text.append(buffer, read);
    return text;
}

void OrderPerThread() {
    std::FILE* file = std::tmpfile();
    Check(file != nullptr, "temporary file");
    auto& log = AgcDriver::AsyncLog::Get();
    std::string expected;
    for (int i = 0; i < 2000; ++i) {
        log.Defer(file, "deferred %d %s %llu\n", i, AgcDriver::LogLiteral{"literal"}, static_cast<unsigned long long>(i) * 3ull);
        expected += "deferred " + std::to_string(i) + " literal " + std::to_string(static_cast<unsigned long long>(i) * 3ull) + "\n";
        const std::string plain = "plain " + std::to_string(i) + "\n";
        log.Write(file, plain.data(), plain.size());
        expected += plain;
    }
    log.Flush();
    Check(ReadAll(file) == expected, "deferred and written lines changed order or content");
    std::fclose(file);
}

void LongLine() {
    std::FILE* file = std::tmpfile();
    Check(file != nullptr, "temporary file");
    const std::string longText(3000, 'x');
    AgcDriver::AsyncLog::Get().Defer(file, "%s|%d\n", AgcDriver::LogLiteral{longText.c_str()}, 7);
    AgcDriver::AsyncLog::Get().Flush();
    Check(ReadAll(file) == longText + "|7\n", "a line longer than the stack buffer changed");
    std::fclose(file);
}

void Coalescing() {
    auto& log = AgcDriver::AsyncLog::Get();
    const std::string line = "[gpu] coalescing test line\n";
    int printed = 0;
    for (int i = 0; i < 50; ++i) printed += log.Admit(line) ? 1 : 0;
    Check(printed == 3, "identical lines in one window were not limited to the first three");
    Check(log.Admit("[gpu] another line\n"), "a different line was suppressed");
}

}

int main() {
    try {
        OrderPerThread();
        LongLine();
        Coalescing();
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
    return 0;
}
