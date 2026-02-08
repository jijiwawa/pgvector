#include "postgres.h"

#include <math.h>

#include "halfutils.h"
#include "neon.h"
#include "halfvec.h"

#ifdef HAS_NEON

/* Vector (FP32) - NEON 优化 */

/* L2 Squared Distance - NEON */
static float
VectorL2SquaredDistanceNeon(int dim, float *ax, float *bx)
{
	float		distance = 0.0f;
	int			i;
	float32x4_t dist_vec = vdupq_n_f32(0.0f);
	int			count = (dim / 4) * 4;

	/* 主循环 - NEON 向量计算 */
	for (i = 0; i < count; i += 4)
	{
		float32x4_t a_vec = vld1q_f32(ax + i);
		float32x4_t b_vec = vld1q_f32(bx + i);
		float32x4_t diff = vsubq_f32(a_vec, b_vec);
		dist_vec = vmlaq_f32(dist_vec, diff, diff);
	}

	/* 水平求和 */
	distance = vaddvq_f32(dist_vec);

	/* 处理剩余元素 */
	for (; i < dim; i++)
	{
		float		diff = ax[i] - bx[i];
		distance += diff * diff;
	}

	return distance;
}

/* Inner Product - NEON */
static float
VectorInnerProductNeon(int dim, float *ax, float *bx)
{
	float		distance = 0.0f;
	int			i;
	float32x4_t sum_vec = vdupq_n_f32(0.0f);
	int			count = (dim / 4) * 4;

	for (i = 0; i < count; i += 4)
	{
		float32x4_t a_vec = vld1q_f32(ax + i);
		float32x4_t b_vec = vld1q_f32(bx + i);
		sum_vec = vmlaq_f32(sum_vec, a_vec, b_vec);
	}

	distance = vaddvq_f32(sum_vec);

	for (; i < dim; i++)
		distance += ax[i] * bx[i];

	return distance;
}

/* Cosine Similarity - NEON */
static double
VectorCosineSimilarityNeon(int dim, float *ax, float *bx)
{
	float		similarity = 0.0f;
	float		norma = 0.0f;
	float		normb = 0.0f;
	int			i;
	float32x4_t sim_vec = vdupq_n_f32(0.0f);
	float32x4_t na_vec = vdupq_n_f32(0.0f);
	float32x4_t nb_vec = vdupq_n_f32(0.0f);
	int			count = (dim / 4) * 4;

	for (i = 0; i < count; i += 4)
	{
		float32x4_t a_vec = vld1q_f32(ax + i);
		float32x4_t b_vec = vld1q_f32(bx + i);
		sim_vec = vmlaq_f32(sim_vec, a_vec, b_vec);
		na_vec = vmlaq_f32(na_vec, a_vec, a_vec);
		nb_vec = vmlaq_f32(nb_vec, b_vec, b_vec);
	}

	similarity = vaddvq_f32(sim_vec);
	norma = vaddvq_f32(na_vec);
	normb = vaddvq_f32(nb_vec);

	for (; i < dim; i++)
	{
		similarity += ax[i] * bx[i];
		norma += ax[i] * ax[i];
		normb += bx[i] * bx[i];
	}

	/* Use sqrt(a * b) over sqrt(a) * sqrt(b) */
	return (double) similarity / sqrt((double) norma * (double) normb);
}

/* L1 Distance - NEON */
static float
VectorL1DistanceNeon(int dim, float *ax, float *bx)
{
	float		distance = 0.0f;
	int			i;
	float32x4_t dist_vec = vdupq_n_f32(0.0f);
	int			count = (dim / 4) * 4;

	for (i = 0; i < count; i += 4)
	{
		float32x4_t a_vec = vld1q_f32(ax + i);
		float32x4_t b_vec = vld1q_f32(bx + i);
		float32x4_t diff = vsubq_f32(a_vec, b_vec);
		dist_vec = vaddq_f32(dist_vec, vabsq_f32(diff));
	}

	distance = vaddvq_f32(dist_vec);

	for (; i < dim; i++)
		distance += fabsf(ax[i] - bx[i]);

	return distance;
}

#endif /* HAS_NEON */

/* SVE 优化 (可选，支持更新的硬件) */

#ifdef HAS_SVE

/* L2 Squared Distance - SVE */
static float
VectorL2SquaredDistanceSve(int dim, float *ax, float *bx)
{
	float		distance = 0.0f;
	svfloat32_t dist_vec = svdup_f32(0.0f);
	int			i = 0;
	svbool_t	pg = svwhilelt_b32(i, dim);

	while (svptest_any(svptrue_b32(), pg))
	{
		svfloat32_t a_vec = svld1_f32(pg, ax + i);
		svfloat32_t b_vec = svld1_f32(pg, bx + i);
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
VectorInnerProductSve(int dim, float *ax, float *bx)
{
	float		distance = 0.0f;
	svfloat32_t sum_vec = svdup_f32(0.0f);
	int			i = 0;
	svbool_t	pg = svwhilelt_b32(i, dim);

	while (svptest_any(svptrue_b32(), pg))
	{
		svfloat32_t a_vec = svld1_f32(pg, ax + i);
		svfloat32_t b_vec = svld1_f32(pg, bx + i);
		sum_vec = svmla_f32_x(pg, sum_vec, a_vec, b_vec);

		i += svcntw();
		pg = svwhilelt_b32(i, dim);
	}

	distance = svaddv_f32(svptrue_b32(), sum_vec);

	return distance;
}

/* Cosine Similarity - SVE */
static double
VectorCosineSimilaritySve(int dim, float *ax, float *bx)
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
		svfloat32_t a_vec = svld1_f32(pg, ax + i);
		svfloat32_t b_vec = svld1_f32(pg, bx + i);
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
VectorL1DistanceSve(int dim, float *ax, float *bx)
{
	float		distance = 0.0f;
	svfloat32_t dist_vec = svdup_f32(0.0f);
	int			i = 0;
	svbool_t	pg = svwhilelt_b32(i, dim);

	while (svptest_any(svptrue_b32(), pg))
	{
		svfloat32_t a_vec = svld1_f32(pg, ax + i);
		svfloat32_t b_vec = svld1_f32(pg, bx + i);
		svfloat32_t diff = svsub_f32_x(pg, a_vec, b_vec);
		dist_vec = svadd_f32_x(pg, dist_vec, svabs_f32_x(pg, diff));

		i += svcntw();
		pg = svwhilelt_b32(i, dim);
	}

	distance = svaddv_f32(svptrue_b32(), dist_vec);

	return distance;
}

#endif /* HAS_SVE */

/* Vector NEON function pointers */
float (*VectorL2SquaredDistanceNeonPtr) (int dim, float *ax, float *bx) = NULL;
float (*VectorInnerProductNeonPtr) (int dim, float *ax, float *bx) = NULL;
double (*VectorCosineSimilarityNeonPtr) (int dim, float *ax, float *bx) = NULL;
float (*VectorL1DistanceNeonPtr) (int dim, float *ax, float *bx) = NULL;

void
NeonVectorInit(void)
{
	/* 优先 SVE (如果支持) */
#ifdef HAS_SVE
	if (SupportsSve())
	{
		VectorL2SquaredDistanceNeonPtr = VectorL2SquaredDistanceSve;
		VectorInnerProductNeonPtr = VectorInnerProductSve;
		VectorCosineSimilarityNeonPtr = VectorCosineSimilaritySve;
		VectorL1DistanceNeonPtr = VectorL1DistanceSve;
		return;
	}
#endif

	/* 回退到 NEON */
#ifdef HAS_NEON
	if (SupportsNeon())
	{
		VectorL2SquaredDistanceNeonPtr = VectorL2SquaredDistanceNeon;
		VectorInnerProductNeonPtr = VectorInnerProductNeon;
		VectorCosineSimilarityNeonPtr = VectorCosineSimilarityNeon;
		VectorL1DistanceNeonPtr = VectorL1DistanceNeon;
	}
#endif
}