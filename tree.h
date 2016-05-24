#include <map>
#include <cmath>
#include "Matrix.h"

struct Key {
  int n;
  int l;
  Key() : n(0), l(0) {}
  Key(int n, int l) : n(n), l(l) {}

  bool operator< (const Key &k) const {
    return ((1<<n)+l < (1<<k.n)+k.l);
  }
  bool operator ==(const Key& k) const {
    return (n == k.n) && (l == k.l);
  }
};

