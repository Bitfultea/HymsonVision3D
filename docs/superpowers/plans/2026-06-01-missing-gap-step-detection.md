# Missing Gap Step Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a missing-gap path so slices with no sensor return inside the gap are measured from the two visible local reference surfaces.

**Architecture:** Detect large x-spacing gaps before the regular fast-path spacing check. Select local support windows adjacent to the gap, fit the existing line/plane pipeline, and keep current absolute/signed height semantics.

**Tech Stack:** C++17, Eigen, OpenCV debug images, existing `bspline_test` self-tests.

---

## File Structure

- Modify `pipeline/GapStepDetection.cpp`: add missing-gap detection helper and call it inside `fast_path_detect_platforms`.
- Modify `test/test_bspline.cpp`: add synthetic missing-gap self-tests.
- No new public API unless a test-only hook is needed under `HYMSON3D_TESTING`.

## Task 1: Add Failing Missing-Gap Self-Test

- [x] Add a synthetic slice in `test/test_bspline.cpp::run_self_test`:
  - left valid surface: `x=0..420`, `z=80 - 0.04*x`
  - missing gap: no points for `x=421..439`
  - right valid surface: `x=440..475`, `z=-45 + 0.03*(x-440)`
  - call `GapStepDetection::test_fast_path_detect_platforms`.

- [x] Assert returned groups:
  - size is `2`
  - left group `x_max >= 415`
  - right group `x_min <= 445`
  - right group span is not a short terminal artifact, e.g. `x_max - x_min >= 20`

- [x] Run:

```bash
cmake --build build --target bspline_test --parallel
./build/test/bspline_test --self-test /tmp/bspline_missing_gap_red
```

Expected: failure because current fast path rejects irregular x-spacing before selecting surfaces.

## Task 2: Implement Missing-Gap Candidate Detection

- [x] In `pipeline/GapStepDetection.cpp`, add an anonymous helper near `fast_path_detect_platforms`:

```cpp
std::vector<std::vector<Eigen::Vector2d>> detect_missing_gap_platforms(
        const std::vector<Eigen::Vector2d>& sorted_pts,
        std::string* fallback_reason);
```

- [x] Inside it:
  - compute positive x-steps and median dx;
  - find the largest x gap;
  - require `max_gap > max(3.0 * median_dx, median_dx + 1e-9)`;
  - split into `left_region` before the gap and `right_region` after the gap;
  - require both sides have at least `kMinSurfacePoints` and enough span.

- [x] Select local reference support:
  - left support = rightmost reliable flat segment before the gap;
  - right support = leftmost reliable flat segment after the gap, expanded only if it remains locally flat;
  - reject if RMS or slope exceeds existing platform limits.

- [x] Return `{left_pts, right_pts}` after `robust_line_fit_inliers`.

## Task 3: Wire Missing-Gap Path Before Regular Spacing Reject

- [x] In `fast_path_detect_platforms`, after sorting `pts` but before the regular spacing validation, call:

```cpp
auto missing_gap_groups = detect_missing_gap_platforms(pts, fallback_reason);
if (!missing_gap_groups.empty()) return missing_gap_groups;
```

- [x] Keep existing regular fast path unchanged when no strong missing gap exists.

- [x] Run the self-test again. Expected: pass.

## Task 4: Add Direction and Equal-Height Coverage

- [x] Add two more synthetic self-test slices:
  - left-low/right-high with missing gap;
  - equal-height left/right with missing gap.

- [x] Verify both return two support groups near the gap. For equal-height, use `test_compute_step_boundaries` if available and assert height is approximately `0`.

## Task 5: Debug and Real-Data Verification

- [x] Run:

```bash
./build/test/bspline_test --self-test /tmp/bspline_missing_gap_self
./build/test/bspline_test --self-test-3d /tmp/bspline_missing_gap_self_3d
./build/test/bspline_test '/home/charles/Data/Dataset/Collected/入壳焊/09_入壳机裁剪图/20260526_172952236_1.tiff' 1,1,100 /tmp/bspline_missing_gap_real
cmake --build build --target pipeline --parallel
git diff --check
```

- [x] Inspect `/tmp/bspline_missing_gap_real/group_pts0.jpg` and a rejected slice. Confirm black endpoints remain on visible surface boundaries, not inside missing data.

## Task 6: Review Criteria

- [x] Missing-gap logic must not trigger on one or two isolated missing pixels.
- [x] Existing non-gap fast path behavior must not regress.
- [x] Width is measured between visible left/right gap boundaries.
- [x] `height_abs` remains non-negative and `signed_height` preserves direction.
- [x] Rejected slice naming still marks abnormal slices.

## Next Planning Step

After this path works on synthetic data, run full-folder statistics on `/home/charles/Data/Dataset/Collected/入壳焊/09_入壳机裁剪图` and compare accepted count, width median, height_abs median, and reject reason distribution against the current baseline.
