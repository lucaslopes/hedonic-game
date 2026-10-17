# The C core alone (make -C native ...): `make test` runs the C tests under
# AddressSanitizer and UndefinedBehaviorSanitizer and the libm check,
# `make test-release` at -O3 as the wheel builds it (one translation unit,
# unity.c, plus unity_fma.c and dispatch.c, which compile to nothing except
# on x86_64 targets without a baseline FMA instruction: core/dispatch.h),
# `make test-soft-fma` at -O3 with the soft fma everywhere
# (HIG_FORCE_SOFT_FMA), `make check-libm` checks that the core calls no libm
# function but sqrt and fabs, and `make test-x86_64` runs all of it as x86_64 under
# Rosetta on Apple Silicon (skipped elsewhere, or when the compiler cannot
# target x86_64). On a dispatch target the tests run as dispatched and then
# with each of the two builds forced. CC=gcc-16 etc. selects the compiler.
CC ?= cc
PYTHON ?= python3
CFLAGS_COMMON = -std=c11 -Wall -Wextra -Wpedantic -Wvla -ffp-contract=off -Iinclude -Icore
SANITIZE = -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined \
  -fno-sanitize-recover=undefined
RELEASE = -O3 -DNDEBUG
SOURCES = $(wildcard core/*/*.c)
HEADERS = include/hedonic_core.h $(wildcard core/*.h core/*/*.h)
# The second build and the entry points of an x86_64 dispatch target.
DISPATCH = unity_fma.c dispatch.c
BUILD = build/c
LEAKS = ASAN_OPTIONS=detect_leaks=$(if $(filter Darwin,$(shell uname)),0,1)
X86 = $(BUILD)/x86_64
# How to run a test program (test-x86_64: arch -x86_64).
RUN =

# $(call each_build,PROGRAM): runs PROGRAM as it dispatches, then with each
# build of a dispatch target forced (HIG_TEST_FMA); test_core.c exits with 77
# when a build cannot be forced (no dispatch here, or no FMA on the CPU).
each_build = $(LEAKS) $(RUN) ./$(1) && \
  for build in soft instruction; do \
    HIG_TEST_FMA=$$build $(LEAKS) $(RUN) ./$(1); status=$$?; \
    if [ $$status -ne 0 ] && [ $$status -ne 77 ]; then exit $$status; fi; \
  done

.PHONY: test test-release test-soft-fma test-x86_64 check-libm clean

test: $(BUILD)/test_core_sanitized check-libm $(BUILD)/test_audit_sanitized
	@$(call each_build,$<)
	$(LEAKS) $(RUN) ./$(filter %test_audit %test_audit_sanitized,$^)

test-release: $(BUILD)/test_core $(BUILD)/test_audit
	@$(call each_build,$<)
	$(LEAKS) $(RUN) ./$(filter %test_audit %test_audit_sanitized,$^)

test-soft-fma: $(BUILD)/test_core_soft $(BUILD)/test_audit_soft
	$(RUN) ./$<
	$(RUN) ./$(BUILD)/test_audit_soft

# The core objects as the wheel compiles them (-O3) and unoptimized: none may
# call a libm function other than sqrt and fabs (tests/check_libm.py says why that
# makes the results the same on every platform).
LIBM_OBJECTS = $(BUILD)/unity_release.o $(BUILD)/unity_debug.o $(BUILD)/unity_fma_release.o \
  $(BUILD)/unity_fma_debug.o $(BUILD)/dispatch_release.o
check-libm: $(LIBM_OBJECTS)
	$(PYTHON) tests/check_libm.py $^

$(BUILD)/%_release.o: %.c unity.c $(SOURCES) $(HEADERS)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS_COMMON) $(RELEASE) -c $< -o $@

$(BUILD)/%_debug.o: %.c unity.c $(SOURCES) $(HEADERS)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS_COMMON) -O0 -c $< -o $@

$(BUILD)/test_core_sanitized: tests/test_core.c $(SOURCES) $(DISPATCH) unity.c $(HEADERS)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS_COMMON) $(SANITIZE) tests/test_core.c $(SOURCES) $(DISPATCH) -lm -o $@

$(BUILD)/test_core: tests/test_core.c unity.c $(DISPATCH) $(SOURCES) $(HEADERS)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS_COMMON) $(RELEASE) tests/test_core.c unity.c $(DISPATCH) -lm -o $@

$(BUILD)/test_core_soft: tests/test_core.c unity.c $(DISPATCH) $(SOURCES) $(HEADERS)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS_COMMON) $(RELEASE) -DHIG_FORCE_SOFT_FMA tests/test_core.c unity.c \
	  $(DISPATCH) -lm -o $@

# x86_64 under Rosetta: the libm check of the x86_64 objects, and the tests,
# sanitized and at -O3, as dispatched and with each build forced, and with
# the soft fma everywhere. Rosetta runs FMA and AVX2 code but reports them in
# CPUID only under ROSETTA_ADVERTISE_AVX=1 (macOS 15 and later; ignored
# before), so the tests run once without it (dispatched to the soft build)
# and once with it (dispatched to the instruction).
test-x86_64:
	@if [ "$$(uname -s)/$$(uname -m)" != "Darwin/arm64" ]; then \
	  echo "test-x86_64: skipped (needs Apple Silicon with Rosetta)"; \
	elif ! arch -x86_64 /usr/bin/true 2>/dev/null; then \
	  echo "test-x86_64: skipped (Rosetta is not installed)"; \
	elif ! (mkdir -p $(BUILD) && echo 'int main(void) { return 0; }' | \
	        $(CC) -arch x86_64 -x c - -o $(BUILD)/arch-probe 2>/dev/null && \
	        [ "$$(lipo -archs $(BUILD)/arch-probe 2>/dev/null)" = x86_64 ]); then \
	  echo "test-x86_64: skipped ($(CC) cannot target x86_64)"; \
	else \
	  $(MAKE) --no-print-directory BUILD=$(X86) CC="$(CC) -arch x86_64" \
	    RUN="arch -x86_64" test test-release test-soft-fma && \
	  $(MAKE) --no-print-directory BUILD=$(X86) CC="$(CC) -arch x86_64" \
	    RUN="env ROSETTA_ADVERTISE_AVX=1 arch -x86_64" test test-release; \
	fi

clean:
	rm -rf $(BUILD)

$(BUILD)/test_audit_sanitized: tests/test_audit.c $(SOURCES) $(DISPATCH) unity.c $(HEADERS)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS_COMMON) $(SANITIZE) $< $(SOURCES) $(DISPATCH) -lm -o $@

$(BUILD)/test_audit: tests/test_audit.c unity.c $(DISPATCH) $(SOURCES) $(HEADERS)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS_COMMON) $(RELEASE) $< unity.c $(DISPATCH) -lm -o $@

$(BUILD)/test_audit_soft: tests/test_audit.c unity.c $(DISPATCH) $(SOURCES) $(HEADERS)
	@mkdir -p $(BUILD)
	$(CC) $(CFLAGS_COMMON) $(RELEASE) -DHIG_FORCE_SOFT_FMA $< unity.c $(DISPATCH) -lm -o $@
