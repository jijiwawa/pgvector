#ifndef NEON_H
#define NEON_H

#include <postgres.h>

/* NEON 检测 */
#if defined(__aarch64__) || defined(__arm__)
#define HAS_NEON
#endif

/* SVE 检测 */
#if defined(__ARM_FEATURE_SVE)
#define HAS_SVE
#endif

#ifdef HAS_NEON
#include <arm_neon.h>
#endif

#ifdef HAS_SVE
#include <arm_sve.h>
#endif

/* Vector NEON/SVE function pointers */
extern float (*VectorL2SquaredDistanceNeonPtr) (int dim, float *ax, float *bx);
extern float (*VectorInnerProductNeonPtr) (int dim, float *ax, float *bx);
extern double (*VectorCosineSimilarityNeonPtr) (int dim, float *ax, float *bx);
extern float (*VectorL1DistanceNeonPtr) (int dim, float *ax, float *bx);

/* CPU feature detection */
#ifdef __aarch64__
static inline bool
SupportsNeon(void)
{
	/* All aarch64 supports NEON */
	return true;
}

static inline bool
SupportsSve(void)
{
#if defined(HAS_SVE)
	return true;
#else
	return false;
#endif
}
#else
static inline bool
SupportsNeon(void)
{
	return false;
}

static inline bool
SupportsSve(void)
{
	return false;
}
#endif

/* Initialization function */
void NeonVectorInit(void);

#endif /* NEON_H */