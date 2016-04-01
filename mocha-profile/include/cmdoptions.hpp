#ifndef _MOCHA_PROFILE_CMDOPTIONS_HPP_
#define _MOCHA_PROFILE_CMDOPTIONS_HPP_

#include "oclobject.hpp"
#include "cmdparser.hpp"

// All command-line options for GEMM sample
class CmdParserMochaGEMM : public CmdParserCommon
{
public:
    // For these options description, please refer to the constructor definition.
    
    // matrix A, B, C have sizes sa, sb, sc.
    CmdOption<size_t> sa, sb, sc;
    CmdOption<int> iterations;
    CmdOption<float> sparsity;

    CmdOption<string> arithmetic;
      CmdEnum<string> arithmetic_float;
      CmdEnum<string> arithmetic_double;
      CmdEnum<string> arithmetic_half;

    CmdOption<string> kernel;

    CmdOption<string> output;

    CmdOption<bool> validation;

    CmdOption<bool> baseline;

    CmdParserMochaGEMM (int argc, const char** argv);

    virtual void parse ();

private:

    template <typename T>
    void validatePositiveness (const CmdOption<T>& parameter)
    {
        parameter.validate(
            parameter.getValue() > 0,
            "negative or zero value is provided; should be positive"
        );
    }

    size_t estimateMaxMatrixSize (
        OpenCLBasic& oclobjects,
        size_t size_of_element,
        size_t alignment
    );
};


#endif  // end of the include guard
