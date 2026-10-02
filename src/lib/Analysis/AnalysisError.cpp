#include <otter/Analysis/AnalysisError.h>

#include <string>

namespace otter {

    namespace {

        class AnalysisErrorCategory final : public std::error_category {
        public:
            const char *name() const noexcept override {
                return "otter.analysis";
            }

            std::string message(int condition) const override {
                switch (static_cast<AnalysisError>(condition)) {
                    case AnalysisError::Cancelled:
                        return cancelledMessage();
                    case AnalysisError::NoWorker:
                        return "no worker thread could be started";
                    case AnalysisError::Internal:
                        return "the analyzer failed internally";
                    case AnalysisError::ContractViolation:
                        return "an object of another contract was produced";
                    case AnalysisError::ModelFailed:
                        return "the model did not produce the expected output";
                    case AnalysisError::ModelMismatch:
                        return "the model produced an output that the variant cannot read";
                }
                return "unknown analysis error";
            }
        };

    }

    const char *cancelledMessage() noexcept {
        return "the execution was cancelled";
    }

    const std::error_category &analysisErrorCategory() noexcept {
        static const AnalysisErrorCategory category;
        return category;
    }

}
