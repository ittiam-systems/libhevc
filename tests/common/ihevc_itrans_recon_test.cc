#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "TestCommon.h"
#include "func_selector.h"
#include "ihevc_defs.h"
#include "ihevc_itrans_utils.h"
#include "ihevc_macros.h"
#include "ihevc_structs.h"
#include "ihevc_typedefs.h"

namespace {

// Test parameters: trans_size, ttype (0: normal, 1: ttype1), arch,
// non_zero_rows, non_zero_cols (number of non-zero rows/columns)
using ITransReconTestParam = std::tuple<int, int, IV_ARCH_T, int, int>;

class ITransReconTest : public ::testing::TestWithParam<ITransReconTestParam> {
 protected:
  void SetUp() override {
    std::tie(trans_size, ttype, arch, num_non_zero_rows, num_non_zero_cols) =
        GetParam();

    src_strd = trans_size;
    pred_strd = trans_size;
    dst_strd = trans_size;

    // TODO: Increase allocations for x86/x86_64 to avoid out-of-bounds
    // reads in SIMD implementations.
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || \
    defined(_M_IX86)
    int pad_pred = (trans_size == 4) ? 8 : 0;
    int pad_tmp = (trans_size == 32) ? 8 : 0;
#else
    int pad_pred = 0;
    int pad_tmp = 0;
#endif

    pi2_src.resize(trans_size * trans_size);
    // pi2_tmp needs to be large enough to hold intermediate data of width *
    // height 16bits.
    pi2_tmp.resize(trans_size * trans_size + pad_tmp);
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || \
    defined(_M_IX86)
    if (trans_size == 32) {
      // SSE4.2 and SSSE3.1 require 3 times trans_size * trans_size for 32x32
      pi2_tmp.resize(3 * trans_size * trans_size + pad_tmp);
    }
#endif
    pu1_pred.resize(trans_size * trans_size + pad_pred);
    pu1_dst_ref.resize(trans_size * trans_size);
    pu1_dst_tst.resize(trans_size * trans_size);

    ref_func_selector = get_ref_func_ptr();
    tst_func_selector = get_tst_func_ptr(arch);
  }

  void RunTest() {
    FillRandomSubBlock(pi2_src.data(), trans_size, src_strd, num_non_zero_rows,
                       num_non_zero_cols, static_cast<WORD16>(-32768),
                       static_cast<WORD16>(32767), 0);
    FillRandom(pu1_pred, static_cast<UWORD8>(0), static_cast<UWORD8>(255), 1);

    WORD32 zero_cols = ComputeZeroMask(trans_size, num_non_zero_cols);
    WORD32 zero_rows = ComputeZeroMask(trans_size, num_non_zero_rows);

    ITransReconFn ref_fn =
        GetITransReconFn(ref_func_selector, trans_size, ttype);
    ITransReconFn tst_fn =
        GetITransReconFn(tst_func_selector, trans_size, ttype);

    ref_fn(pi2_src.data(), pi2_tmp.data(), pu1_pred.data(), pu1_dst_ref.data(),
           src_strd, pred_strd, dst_strd, zero_cols, zero_rows);
    tst_fn(pi2_src.data(), pi2_tmp.data(), pu1_pred.data(), pu1_dst_tst.data(),
           src_strd, pred_strd, dst_strd, zero_cols, zero_rows);
    ASSERT_NO_FATAL_FAILURE(compare_output<UWORD8>(
        pu1_dst_ref, pu1_dst_tst, trans_size, trans_size, dst_strd));
  }

  int trans_size;
  int ttype;
  IV_ARCH_T arch;
  const ihevc_func_selector_t* ref_func_selector;
  const ihevc_func_selector_t* tst_func_selector;

  WORD32 src_strd;
  WORD32 pred_strd;
  WORD32 dst_strd;
  WORD32 num_non_zero_rows;
  WORD32 num_non_zero_cols;
  std::vector<WORD16> pi2_src;
  std::vector<WORD16> pi2_tmp;
  std::vector<UWORD8> pu1_pred;
  std::vector<UWORD8> pu1_dst_ref;
  std::vector<UWORD8> pu1_dst_tst;
};

TEST_P(ITransReconTest, Run) { RunTest(); }

std::string PrintITransReconTestParam(
    const testing::TestParamInfo<ITransReconTestParam> &info) {
  WORD32 trans_size, ttype, non_zero_rows, non_zero_cols;
  IV_ARCH_T arch;
  std::tie(trans_size, ttype, arch, non_zero_rows, non_zero_cols) = info.param;
  std::stringstream ss;
  ss << "size_" << trans_size << "_ttype_" << ttype << "_nzr_" << non_zero_rows
     << "_nzc_" << non_zero_cols << "_" << get_arch_str(arch);
  return ss.str();
}

std::vector<ITransReconTestParam> GenerateITransReconTestParams() {
  std::vector<ITransReconTestParam> params;
  const WORD32 nz_options[] = {1, 2, 4, 8, 16, 32};

  auto add_params_for_size = [&](int size, const int *ttypes, int num_ttypes) {
    for (int t = 0; t < num_ttypes; t++) {
      int ttype = ttypes[t];
      for (auto arch : getTstArch()) {
        for (WORD32 nnzr : nz_options) {
          if (nnzr > size)
            continue;
          for (WORD32 nnzc : nz_options) {
            if (nnzc > size)
              continue;
            params.emplace_back(size, ttype, arch, nnzr, nnzc);
          }
        }
      }
    }
  };

  const int ttypes4[] = {0, 1};
  const int ttypesOther[] = {0};

  add_params_for_size(4, ttypes4, 2);
  add_params_for_size(8, ttypesOther, 1);
  add_params_for_size(16, ttypesOther, 1);
  add_params_for_size(32, ttypesOther, 1);

  return params;
}

INSTANTIATE_TEST_SUITE_P(ITransRecon, ITransReconTest,
                         ::testing::ValuesIn(GenerateITransReconTestParams()),
                         PrintITransReconTestParam);

} // namespace
