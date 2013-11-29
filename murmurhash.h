/*
 * Murmur Hash is in the public domain. This version from Peter Scott,
 * translated from the original version from Austin Appleby.
 */

#ifndef MURMURHASH_H_
#define MURMURHASH_H_

#include <stdint.h>

void MurmurHash3_x64_128(const void *key, const int len, const uint32_t seed, void *out);

#endif  // MURMURHASH_H_
