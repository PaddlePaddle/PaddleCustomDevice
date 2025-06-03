/*
 * math_constants.h -
 *  MACA equivalent of the CUDA header of the same name
 */

#ifndef __MATH_CONSTANTS_H__
#define __MATH_CONSTANTS_H__

/* single precision constants */

#define MACART_INF_F        __int_as_float(0x7f800000)
#define MACART_NAN_F        __int_as_float(0x7fffffff)
#define MACART_MIN_DENORM_F __int_as_float(0x00000001)
#define MACART_MAX_NORMAL_F __int_as_float(0x7f7fffff)
#define MACART_NEG_ZERO_F   __int_as_float(0x80000000)
#define MACART_ZERO_F       0.0f
#define MACART_ONE_F        1.0f

/* double precision constants */
#define MACART_INF          __hiloint2double(0x7ff00000, 0x00000000)
#define MACART_NAN          __hiloint2double(0xfff80000, 0x00000000)

#endif
