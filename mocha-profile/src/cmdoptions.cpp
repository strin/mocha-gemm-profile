#include <limits>
#include <cmath>

#include "cmdoptions.hpp"

using namespace std;


#ifdef _MSC_VER
#pragma warning (push)
#pragma warning (disable : 4355)    // 'this': used in base member initializer list
#endif

CmdParserMochaGEMM::CmdParserMochaGEMM (int argc, const char** argv) :
    CmdParserCommon(argc, argv),
    sa(
        *this,
        'a',
        "sa",
        "<integer>",
        "#row of matrix A",
        1024
    ),
    sb(
        *this,
        'b',
        "sb",
        "<integer>",
        "#column of matrix A and #row of matrix B",
        1024
    ),
    sc(
        *this,
        'c',
        "sc",
        "<integer>",
        "#column of matrix B",
        1024
    ),
    iterations(
        *this,
        'i',
        "iterations",
        "<integer>",
        "Number of kernel invocations. For each invoction, "
            "performance information will be printed. "
            "Zero is allowed: in this case no kernel invocation "
            " is performed but all other host stuff is created.",
        10
    ),
    arithmetic(
        *this,
        'a',
        "arithmetic",
        "",
        "Type of elements and all calculations.",
        "float"
    ),
    arithmetic_float(arithmetic, "float"),
    arithmetic_double(arithmetic, "double"),
    arithmetic_half(arithmetic, "half"),
    kernel(
        *this,
        0,
        "kernel",
        "",
        "Determines format of matrices involved in multiplication. "
            "There are two supported form: nn and nt; nn is for case when "
            "both matrices A and B are in column-major form; nt is for case "
            "when A is in column-major form, but B is in row major format "
            "(i.e. transposed). Matrices A and C are always in column major "
            "format.",
        "nn"
    ),
    validation(
        *this,
        0,
        "validation",
        "",
        "Enables validation procedure on host (slow for big matrices).",
        false
    ),
    output(
        *this,
        'o',
        "output",
        "",
        "The file to which output is written.",
        "result.json"
    )
{
}

#ifdef _MSC_VER
#pragma warning (pop)
#endif

void CmdParserMochaGEMM::parse ()
{
    CmdParserCommon::parse();
}


size_t CmdParserMochaGEMM::estimateMaxMatrixSize (
    OpenCLBasic& oclobjects,
    size_t size_of_element,
    size_t alignment
)
{
    cl_ulong max_alloc_size = 0;
    cl_int err = clGetDeviceInfo(
        oclobjects.device,
        CL_DEVICE_MAX_MEM_ALLOC_SIZE,
        sizeof(max_alloc_size),
        &max_alloc_size,
        0
    );
    SAMPLE_CHECK_ERRORS(err);

    cl_ulong max_global_mem_size = 0;
    err = clGetDeviceInfo(
        oclobjects.device,
        CL_DEVICE_GLOBAL_MEM_SIZE,
        sizeof(max_global_mem_size),
        &max_global_mem_size,
        0
    );
    SAMPLE_CHECK_ERRORS(err);

    double max_matrix_size = sqrt(
        min(
            double(numeric_limits<size_t>::max()),
            min(double(max_alloc_size), double(max_global_mem_size)/3)
        ) / size_of_element
    );

    assert(alignment%size_of_element == 0);

    // the following is effect of a bit conservative
    // estimation of the overhead on a row alignment
    max_matrix_size -= alignment/size_of_element;

    assert(max_matrix_size < double(numeric_limits<size_t>::max()));

    return static_cast<size_t>(max_matrix_size);
}

