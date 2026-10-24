#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_SHADERPREPARATION_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_SHADERPREPARATION_HPP

#include <memory>

namespace AgcDriver::DriverDetail {

struct ShaderSnapshot;
struct PreparedShaderState;

class ShaderPreparationTransaction {
public:
    ShaderPreparationTransaction();
    ~ShaderPreparationTransaction();
    ShaderPreparationTransaction(const ShaderPreparationTransaction&) = delete;
    ShaderPreparationTransaction& operator=(const ShaderPreparationTransaction&) = delete;
    PreparedShaderState& Edit(const ShaderSnapshot& snapshot);
    const PreparedShaderState& Read(const ShaderSnapshot& snapshot) const;
    bool Tracks(const ShaderSnapshot& snapshot) const;
    void Hold();
    void Commit();
    // A refused preparation was skipped (APS5_SKIP_REFUSED_SHADERS): the transactions it unwound do not
    // abort the one still open on this thread, which keeps and commits the work done before it.
    static void ContinueAfterRefusal();

private:
    struct State;
    std::unique_ptr<State> state;
    State* root = nullptr;
    bool committed = false;
};

}

#endif
