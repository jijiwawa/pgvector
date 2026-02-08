#include "postgres.h"

#include <math.h>

#include "halfutils.h"
#include "neon.h"
#include "halfvec.h"

#ifdef HAS_NEON

/* HalfVector (FP16) - NEON 优化 */

/* L2 Squared Distance - NEON (FP16 直接运算) */
static float
HalfvecL2SquaredDistanceNeon(int dim, half * ax, half * bx)
{
	float		distance = 0.0f;
	int			i;
	float32x4_t dist_vec = vdupq_n_f32(0.0f);
	int			count = (dim / 8) * 8;

	/* 使用 FP16 直接加载，避免转换开销 */
	for (i = 0; i < count; i += 8)
	{
		float16x8_t a_vec = vld1q_f16((float16_t *) (ax + i));
		float16x8_t b_vec = vld1q_f16((float16_t *) (bx + i));

		/* 转换到 FP32 进行精确计算 */
		float32x4_t a_lo = vcvt_f32_f16(vget_low_f16(a_vec));
		float32x4_t a_hi = vcvt_f32_f16(vget_high_f16(a_vec));
		float32x4_t b_lo = vcvt_f32_f16(vget_low_f16(b_vec));
		float32x4_t b_hi = vcvt_f32_f16(vget_high_f16(b_vec));

		float32x4_t diff_lo = vsubq_f32(a_lo, b_lo);
		float32x4_t diff_hi = vsubq_f32(a_hi, b_hi);

		dist_vec = vmlaq_f32(dist_vec, diff_lo, diff_lo);
		dist_vec = vmlaq_f32(dist_vec, diff_hi, diff_hi);
	}

	distance = vaddvq_f32(dist_vec);

	for (; i < dim; i++)
	{
		float		diff = HalfToFloat4(ax[i]) - HalfToFloat4(bx[i]);
		distance += diff * diff;
	}

	return distance;
}

/* Inner Product - NEON */
static float
HalfvecInnerProductNeon(int dim, half * ax, half * bx)
{
	float		distance = 0.0f;
	int			i;
	float32x4_t sum_vec = vdupq_n_f32(0.0f);
	int			count = (dim / 8) * 8;

	for (i = 0; i < count; i += 8)
	{
		float16x8_t a_vec = vld1q_f16((float16_t *) (ax + i));
		float16x8_t b_vec = vld1q_f16((float16_t *) (bx + i));

		float32x4_t a_lo = vcvt_f32_f16(vget_low_f16(a_vec));
		float32x4_t a_hi = vcvt_f32_f16(vget_high_f16(a_vec));
		float32x4_t b_lo = vcvt_f32_f16(vget_low_f16(b_vec));
		float32x4_t b_hi = vcvt_f32_f16(vget_high_f16(b_vec));

		sum_vec = vmlaq_f32(sum_vec, a_lo, b_lo);
		sum_vec = vmlaq_f32(sum_vec, a_hi, b_hi);
	}

	distance = vaddvq_f32(sum_vec);

	for (; i < dim; i++)
		distance += HalfToFloat4(ax[i]) * HalfToFloat4(bx[i]);

	return distance;
}

/* Cosine Similarity - NEON */
static double
HalfvecCosineSimilarityNeon(int dim, half * ax, half * bx)
{
	float		similarity = 0.0f;
	float		norma = 0.0f;
	float		normb = 0.0f;
	int			i;
	float32x4_t sim_vec = vdupq_n_f32(0.0f);
	float32x4_t na_vec = vdupq_n_f32(0.0f);
	float32x4_t nb_vec = vdupq_n_f32(0.0f);
	int			count = (dim / 8) * 8;

	for (i = 0; i < count; i += 8)
	{
		float16x8_t a_vec = vld1q_f16((float16_t *) (ax + i));
		float16x8_t b_vec = vld1q_f16((float16_t *) (bx + i));

		float32x4_t a_lo = vcvt_f32_f16(vget_low_f16(a_vec));
		float32x4_t a_hi = vcvt_f32_f16(vget_high_f16(a_vec));
		float32x4_t b_lo = vcvt_f32_f16(vget_low_f16(b_vec));
		float32x4_t b_hi = vcvt_f32_f16(vget_high_f16(b_vec));

		sim_vec = vmlaq_f32(sim_vec, a_lo, b_lo);
		sim_vec = vmlaq_f32(sim_vec, a_hi, b_hi);
		na_vec = vmlaq_f32(na_vec, a_lo, a_lo);
		na_vec = vmlaq_f32(na_vec, a_hi, a_hi);
		nb_vec = vmlaq_f32(nb_vec, b_lo, b_lo);
		nb_vec = vmlaq_f32(nb_vec, b_hi, b_hi);
	}

	similarity = vaddvq_f32(sim_vec);
	norma = vaddvq_f32(na_vec);
	normb = vaddvq_f32(nb_vec);

	for (; i < dim; i++)
	{
		float		axi = HalfToFloat4(ax[i]);
		float		bxi = HalfToFloat4(bx[i]);
		similarity += axi * bxi;
		norma += axi * axi;
		normb += bxi * bxi;
	}

	/* Use sqrt(a * b) over sqrt(a) * sqrt(b) */
	return (double) similarity / sqrt((double) norma * (double) normb);
}

/* L1 Distance - NEON */
static float
HalfvecL1DistanceNeon(int dim, half * ax, half * bx)
{
	float		distance = 0.0f;
	int			i;
	float32x4_t dist_vec = vdupq_n_f32(0.0f);
	int			count = (dim / 8) * 8;

	for (i = 0; i < count; i += 8)
	{
		float16x8_t a_vec = vld1q_f16((float16_t *) (ax + i));
		float16x8_t b_vec = vld1q_f16((float16_t *) (bx + i));

		float32x4_t a_lo = vcvt_f32_f16(vget_low_f16(a_vec));
		float32x4_t a_hi = vcvt_f32_f16(vget_high_f16(a_vec));
		float32x4_t b_lo = vcvt_f32_f16(vget_low_f16(b_vec));
		float32x4_t b_hi = vcvt_f32_f16(vget_high_f16(b_vec));

		float32x4_t diff_lo = vsubq_f32(a_lo, b_lo);
		float32x4_t diff_hi = vsubq_f32(a_hi, b_hi);

		dist_vec = vaddq_f32(dist_vec, vabsq_f32(diff_lo));
		dist_vec = vaddq_f32(dist_vec, vabsq_f32(diff_hi));
	}

	distance = vaddvq_f32(dist_vec);

	for (; i < dim; i++)
		distance += fabsf(HalfToFloat4(ax[i]) - HalfToFloat4(bx[i]));

	return distance;
}

#endif /* HAS_NEON */

/* SVE 优化 (可选，支持更新的硬件) */

#ifdef HAS_SVE

/* L2 Squared Distance - SVE */
static float
HalfvecL2SquaredDistanceSve(int dim, half * ax, half * bx)
{
	float		distance = 0.0f;
	svfloat32_t dist_vec = svdup_f32(0.0f);
	int			i = 0;
	svbool_t	pg = svwhilelt_b32(i, dim);

	while (svptest_any(svptrue_b32(), pg))
	{
		/* SVE 不直接支持 FP16，使用 FP32 计算 */
		svfloat32_t a_vec = svld1_f32(pg, (float *) (ax + i));
		svfloat32_t b_vec = svld1_f32(pg, (float *) (bx + i));
		svfloat32_t diff = svsub_f32_x(pg, a_vec, b_vec);
		dist_vec = svmla_f32_x(pg, dist_vec, diff, diff);

		i += svcntw();
		pg = svwhilelt_b32(i, dim);
	}

	distance = svaddv_f32(svptrue_b32(), dist_vec);

	return distance;
}

/* Inner Product - SVE */
static float
HalfvecInnerProductSve(int dim, half * ax, half * bx)
{
	float		distance = 0.0f;
	svfloat32_t sum_vec = svdup_f32(0.0f);
	int			i = 0;
	svbool_t	pg = svwhilelt_b32(i, dim);

	while (svptest_any(svptrue_b32(), pg))
	{
		svfloat32_t a_vec = svld1_f32(pg, (float *) (ax + i));
		svfloat32_t b_vec = svld1_f32(pg, (float *) (bx + i));
		sum_vec = svmla_f32_x(pg, sum_vec, a_vec, b_vec);

		i += svcntw();
		pg = svwhilelt_b32(i, dim);
	}

	distance = svaddv_f32(svptrue_b32(), sum_vec);

	return distance;
}

/* Cosine Similarity - SVE */
static double
HalfvecCosineSimilaritySve(int dim, half * ax, half * bx)
{
	float		similarity = 0.0f;
	float		norma = 0.0f;
	float		normb = 0.0f;
	svfloat32_t sim_vec = svdup_f32(0.0f);
	svfloat32_t na_vec = svdup_f32(0.0f);
	svfloat32_t nb_vec = svdup_f32(0.0f);
	int			i = 0;
	svbool_t	pg = svwhilelt_b32(i, dim);

	while (svptest_any(svptrue_b32(), pg))
	{
		svfloat32_t a_vec = svld1_f32(pg, (float *) (ax + i));
		svfloat32_t b_vec = svld1_f32(pg, (float *) (bx + i));
		sim_vec = svmla_f32_x(pg, sim_vec, a_vec, b_vec);
		na_vec = svmla_f32_x(pg, na_vec, a_vec, a_vec);
		nb_vec = svmla_f32_x(pg, nb_vec, b_vec, b_vec);

		i += svcntw();
		pg = svwhilelt_b32(i, dim);
	}

	similarity = svaddv_f32(svptrue_b32(), sim_vec);
	norma = svaddv_f32(svptrue_b32(), na_vec);
	normb = svaddv_f32(svptrue_b32(), nb_vec);

	return (double) similarity / sqrt((double) norma * (double) normb);
}

/* L1 Distance - SVE */
static float
HalfvecL1DistanceSve(int dim, half * ax, half * bx)
{
	float		distance = 0.0f;
	svfloat32_t dist_vec = svdup_f32(0.0f);
	int			i = 0;
	svbool_t	pg = svwhilelt_b32(i, dim);

	while (svptest_any(svptrue_b32(), pg))
	{
		svfloat32_t a_vec = svld1_f32(pg, (float *) (ax + i));
		svfloat32_t b_vec = svld1_f32(pg, (float *) (bx + i));
		svfloat32_t diff = svsub_f32_x(pg, a_vec, b_vec);
		dist_vec = svadd_f32_x(pg, dist_vec, svabs_f32_x(pg, diff));

		i += svcntw();
		pg = svwhilelt_b32(i, dim);
	}

	distance = svaddv_f32(svptrue_b32(), dist_vec);

	return distance;
}

#endif /* HAS_SVE */

/* HalfVector NEON 函数指针 - 导出 */
float	   (*HalfvecL2SquaredDistanceNeonPtr) (int dim, half * ax, half * bx) = NULL;
float	   (*HalfvecInnerProductNeonPtr) (int dim, half * ax, half * bx) = NULL;
double	  (*HalfvecCosineSimilarityNeonPtr) (int dim, half * ax, half * bx) = NULL;
float	   (*HalfvecL1DistanceNeonPtr) (int dim, half * ax, half * bx) = NULL;

/* HalfVector SVE 函数指针 - 导出 */
float	   (*HalfvecL2SquaredDistanceSvePtr) (int dim, half * ax, half * bx) = NULL;
float	   (*HalfvecInnerProductSvePtr) (int dim, half * ax, half * bx) = NULL;
double	  (*HalfvecCosineSimilaritySvePtr) (int dim, half * ax, half * bx) = NULL;
float	   (*HalfvecL1DistanceSvePtr) (int dim, half * ax, half * bx) = NULL;

void
NeonHalfvecInit(void)
{
	/* 优先 SVE (如果支持) */
#ifdef HAS_SVE
	if (SupportsSve())
	{
		HalfvecL2SquaredDistanceNeonPtr = HalfvecL2SquaredDistanceSve;
		HalfvecInnerProductNeonPtr = HalfvecInnerProductSve;
		HalfvecCosineSimilarityNeonPtr = HalfvecCosineSimilaritySve;
		HalfvecL1DistanceNeonPtr = HalfvecL1DistanceSve;
		return;
	}
#endif

	/* 回退到 NEON */
#ifdef HAS_NEON
	if (SupportsNeon())
	{
		HalfvecL2SquaredDistanceNeonPtr = HalfvecL2SquaredDistanceNeon;
		HalfvecInnerProductNeonPtr = HalfvecInnerProductNeon;
		HalfvecCosineSimilarityNeonPtr = HalfvecCosineSimilarityNeon;
		HalfvecL1DistanceNeonPtr = HalfvecL1DistanceNeon;
	}
#endif
}
