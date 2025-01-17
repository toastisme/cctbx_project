#ifndef KOKKOSTBX_MATRIX3_H
#define KOKKOSTBX_MATRIX3_H

#include <cmath>
#include <iostream>
#include <Kokkos_Core.hpp>

namespace kokkostbx {

    template <typename NumType>
    struct matrix3 {

        NumType data[9] = { };

        // CONSTRUCTOR
        KOKKOS_FUNCTION matrix3() = default;

        KOKKOS_FUNCTION matrix3(NumType a) : {
            for (NumType& d : data) {
                d = a;
            }
        }

        KOKKOS_FUNCTION matrix3(NumType a[9]) : data(a[0]), y(a[1]), z(a[2]) {}

        KOKKOS_FUNCTION matrix3(NumType a, NumType b, NumType c) : x(a), y(b), z(c) {}

        // OPERATORS
        // streaming
        friend std::ostream& operator<< (std::ostream &os, const matrix3<NumType>& v) {
            os << v.x << ", " << v.y << ", " << v.z;
            return os;
        }

        // access
        KOKKOS_INLINE_FUNCTION NumType& operator[](const int index) {
            switch(index) {
                case 0: return x;
                case 1: return y;
                case 2: return z;
                default:
                    ::Kokkos::abort("Can't access kokkos::matrix3, index outside [0,1,2].");
            }
        }

        // addition
        KOKKOS_INLINE_FUNCTION friend matrix3<NumType> operator+(const matrix3<NumType>& lhs, const matrix3<NumType>& rhs) {
            return matrix3<NumType>(lhs.x+rhs.x, lhs.y+rhs.y, lhs.z+rhs.z);
        }
        KOKKOS_INLINE_FUNCTION friend matrix3<NumType> operator+(const matrix3<NumType>& lhs, NumType& rhs) {
            return matrix3<NumType>(lhs.x+rhs, lhs.y+rhs, lhs.z+rhs);
        }
        KOKKOS_INLINE_FUNCTION friend matrix3<NumType> operator+(NumType lhs, const matrix3<NumType>& rhs) {
            return rhs + lhs;
        }

        KOKKOS_INLINE_FUNCTION void operator+=(const matrix3<NumType>& v) {
            x += v.x;
            y += v.y;
            z += v.z;
        }

        KOKKOS_INLINE_FUNCTION void operator+=(const NumType& v) {
            x += v;
            y += v;
            z += v;
        }

        // subtraction
        KOKKOS_INLINE_FUNCTION matrix3<NumType> operator-() const {
            return matrix3<NumType>(-x, -y, -z);
        }

        KOKKOS_INLINE_FUNCTION matrix3<NumType> operator-(const matrix3<NumType>& v) const {
            return matrix3<NumType>(x-v.x, y-v.y, z-v.z);
        }

        KOKKOS_INLINE_FUNCTION matrix3<NumType> operator-(const NumType& v) const {
            return matrix3<NumType>(x-v, y-v, z-v);
        }

        KOKKOS_INLINE_FUNCTION void operator-=(const matrix3<NumType>& v) {
            x -= v.x;
            y -= v.y;
            z -= v.z;
        }

        KOKKOS_INLINE_FUNCTION void operator-=(const NumType& v) {
            x -= v;
            y -= v;
            z -= v;
        }

        // multiplication
        KOKKOS_INLINE_FUNCTION friend matrix3<NumType> operator*(const matrix3<NumType>& lhs, const matrix3<NumType>& rhs) {
            return matrix3<NumType>(lhs.x*rhs.x, lhs.y*rhs.y, lhs.z*rhs.z);
        }
        KOKKOS_INLINE_FUNCTION friend matrix3<NumType> operator*(const matrix3<NumType>& lhs, NumType& rhs) {
            return matrix3<NumType>(lhs.x*rhs, lhs.y*rhs, lhs.z*rhs);
        }
        KOKKOS_INLINE_FUNCTION friend matrix3<NumType> operator*(NumType lhs, const matrix3<NumType>& rhs) {
            return rhs * lhs;
        }

        KOKKOS_INLINE_FUNCTION void operator*=(const matrix3<NumType>& v) {
            x *= v.x;
            y *= v.y;
            z *= v.z;
        }

        KOKKOS_INLINE_FUNCTION void operator*=(const NumType& v) {
            x *= v;
            y *= v;
            z *= v;
        }

        // division
        KOKKOS_INLINE_FUNCTION matrix3<NumType> operator/(const matrix3<NumType>& v) const {
            return matrix3<NumType>(x/v.x, y/v.y, z/v.z);
        }

        KOKKOS_INLINE_FUNCTION matrix3<NumType> operator/(const NumType& v) const {
            return matrix3<NumType>(x/v, y/v, z/v);
        }

        KOKKOS_INLINE_FUNCTION void operator/=(const matrix3<NumType>& v) {
            x /= v.x;
            y /= v.y;
            z /= v.z;
        }

        KOKKOS_INLINE_FUNCTION void operator/=(const NumType& v) {
            x /= v;
            y /= v;
            z /= v;
        }

        // METHODS
        KOKKOS_INLINE_FUNCTION void zero() {
            x = 0;
            z = 0;
            y = 0;
        }

        KOKKOS_INLINE_FUNCTION NumType length_sqr() const {
            return x*x + y*y + z*z;
        }

        KOKKOS_INLINE_FUNCTION NumType length() const {
            return ::Kokkos::Experimental::sqrt(length_sqr());
        }

        KOKKOS_INLINE_FUNCTION NumType dot(const matrix3<NumType>& v) const {
            return x*v.x + y*v.y + z*v.z;
        }

        KOKKOS_INLINE_FUNCTION matrix3<NumType> cross(const matrix3<NumType>& v) const {
            return matrix3<NumType>(y*v.z - z*v.y,
                                    z*v.x - x*v.z,
                                    x*v.y - y*v.x);
        }

        KOKKOS_INLINE_FUNCTION void normalize() {
            NumType l = length();
            if (l>0) {
                x /= l;
                y /= l;
                z /= l;
            }
        }

        KOKKOS_INLINE_FUNCTION matrix3<NumType> get_unit_vector() const {
            NumType l = length();
            if (l>0) {
                return matrix3<NumType>(x/l, y/l, z/l);
            } else {
                return matrix3<NumType>(0, 0, 0);
            }
        }



}

#endif
