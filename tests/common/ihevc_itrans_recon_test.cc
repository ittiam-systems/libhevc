#include <algorithm>
#include <gtest/gtest.h>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "func_selector.h"
#include "ihevc_defs.h"
#include "ihevc_itrans_recon.h"
#include "ihevc_macros.h"
#include "ihevc_structs.h"
#include "ihevc_typedefs.h"
#include "tests_common.h"

namespace {

// Test parameters: trans_size, ttype (0: normal, 1: ttype1), arch,
// non_zero_rows, non_zero_cols (number of non-zero rows/columns)
using ITransReconTestParam = std::tuple<int, int, IVD_ARCH_T, int, int>;

class ITransReconTest : public ::testing::TestWithParam<ITransReconTestParam> {
protected:
  void SetUp() override {
    std::tie(trans_size, ttype, arch, num_non_zero_rows, num_non_zero_cols) =
        GetParam();

    src_strd = trans_size;
    pred_strd = trans_size;
    dst_strd = trans_size;

    pi2_src.resize(trans_size * trans_size);
    // pi2_tmp needs to be large enough to hold intermediate data of width *
    // height 16bits.
    pi2_tmp.resize(trans_size * trans_size);
    pu1_pred.resize(trans_size * trans_size);
    pu1_dst_ref.resize(trans_size * trans_size);
    pu1_dst_tst.resize(trans_size * trans_size);

    ref_func_selector = get_ref_func_ptr();
    tst_func_selector = get_tst_func_ptr(arch);
  }

  template <typename FuncPtr> void RunTest(FuncPtr func_ptr) {
    if (ref_func_selector == nullptr || tst_func_selector == nullptr) {
      GTEST_SKIP() << "Function selector not available";
    }

    std::mt19937 rng(0);
    std::uniform_int_distribution<int16_t> coeff_dist(-32768, 32767);
    std::uniform_int_distribution<uint8_t> pixel_dist(0, 255);

    // Populate pi2_src so that the requested number of rows and columns
    // are potentially non-zero. Rows [0, non_zero_rows) and columns
    // [0, non_zero_cols) form the non-zero region; everything else is zero.
    std::fill(pi2_src.begin(), pi2_src.end(), 0);
    for (int i = 0; i < trans_size; i++) {
      for (int j = 0; j < trans_size; j++) {
        if (i < num_non_zero_rows && j < num_non_zero_cols) {
          pi2_src[i * src_strd + j] = coeff_dist(rng);
        }
      }
    }

    for (auto &v : pu1_pred)
      v = pixel_dist(rng);

    WORD32 non_zero_rows_mask = 0;
    for (int i = 0; i < num_non_zero_rows && i < trans_size; i++) {
      non_zero_rows_mask |= (1u << i);
    }

    WORD32 non_zero_cols_mask = 0;
    for (int j = 0; j < num_non_zero_cols && j < trans_size; j++) {
      non_zero_cols_mask |= (1u << j);
    }

    WORD32 mask = (trans_size == 32)
                      ? 0xFFFFFFFFu
                      : ((static_cast<WORD32>(1u) << trans_size) - 1u);
    WORD32 zero_cols = (~non_zero_cols_mask) & mask;
    WORD32 zero_rows = (~non_zero_rows_mask) & mask;

    (ref_func_selector->*func_ptr)(
        pi2_src.data(), pi2_tmp.data(), pu1_pred.data(), pu1_dst_ref.data(),
        src_strd, pred_strd, dst_strd, zero_cols, zero_rows);
    (tst_func_selector->*func_ptr)(
        pi2_src.data(), pi2_tmp.data(), pu1_pred.data(), pu1_dst_tst.data(),
        src_strd, pred_strd, dst_strd, zero_cols, zero_rows);
    ASSERT_NO_FATAL_FAILURE(compare_output<UWORD8>(
        pu1_dst_ref, pu1_dst_tst, trans_size, trans_size, dst_strd));
  }

  int trans_size;
  int ttype;
  IVD_ARCH_T arch;
  const func_selector_t *ref_func_selector;
  const func_selector_t *tst_func_selector;

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

TEST_P(ITransReconTest, Run) {
  if (trans_size == 4) {
    if (ttype == 1) {
      RunTest(&func_selector_t::ihevc_itrans_recon_4x4_ttype1_fptr);
    } else {
      RunTest(&func_selector_t::ihevc_itrans_recon_4x4_fptr);
    }
  } else if (trans_size == 8) {
    RunTest(&func_selector_t::ihevc_itrans_recon_8x8_fptr);
  } else if (trans_size == 16) {
    RunTest(&func_selector_t::ihevc_itrans_recon_16x16_fptr);
  } else if (trans_size == 32) {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) ||               \
    defined(_M_IX86)
    GTEST_SKIP() << "SSE4.2 and SSSE3 are not matching C implementation for "
                    "ihevc_itrans_recon_32x32_fptr";
#endif
    RunTest(&func_selector_t::ihevc_itrans_recon_32x32_fptr);
  }
}

std::string PrintITransReconTestParam(
    const testing::TestParamInfo<ITransReconTestParam> &info) {
  WORD32 trans_size, ttype, non_zero_rows, non_zero_cols;
  IVD_ARCH_T arch;
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
      for (auto arch : ga_tst_arch) {
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
