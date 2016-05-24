#include <madness/world/MADworld.h>
#include <madness/world/world.h>
#include <madness/world/worlddc.h>
#include "twoscale.h"
#include "gauss_legendre.h"

using namespace madness;

Tensor<double> zeros(int s0) {
  Tensor<double> r(s0*1L);
  return r;
}

Tensor<double> zeros(int s0, int s1) {
  Tensor<double> r(s0*1L, s1*1L);
  return r;
}

double normf(Tensor<double> t) {
  return t.normf();
}

double wst_inner(const Tensor<double>& t1, const Tensor<double>& t2) {
  double rval = 0.0;
  for (auto i = 0; i < t1.size(); i++) rval += t1[i]*t2[i];
  return rval;
}

struct Key {
  int n;
  int l;
  Key() : n(0), l(0) {}
  Key(int n, int l) : n(n), l(l) {}

  hashT hash() const {
      hashT h = hash_value(n);
      hash_combine(h,l);
      return h;
  }

  bool operator< (const Key &k) const {
    return ((1<<n)+l < (1<<k.n)+k.l);
  }
  bool operator ==(const Key& k) const {
    return (n == k.n) && (l == k.l);
  }
  template <typename Archive>
  void serialize(const Archive& ar) {
      ar & n & l;
  }
  Key left_child() const {
    return Key(n+1, 2*l);
  }
  Key right_child() const {
    return Key(n+1, 2*l+1);
  }
};

struct Node {
  Key key;
  Tensor<double> coeffs;
  bool has_children;
  // Node() : key(), coeffs(), has_children(false) {}
  // Node(const Key& key, const Tensor<double>& coeffs, bool has_children) 
  //  : key(key), coeffs(coeffs), has_children(has_children) {}
  Node() : key(), coeffs() {}
  Node(const Key& key, const Tensor<double>& coeffs) 
   : key(key), coeffs(coeffs) {}

  bool operator ==(const Node& a) const {
    return (key == a.key) && ((coeffs - a.coeffs).normf() < 1e-12);
  }
  bool operator !=(const Node& a) const {
    return !(*this==a);
  }
  template <typename Archive>
  void serialize(const Archive& ar) {
    //ar & key & coeffs & has_children;
    ar & key & coeffs;
  }
};

double normf(const Node& node) {
  return node.coeffs.normf();
}

using CoeffTree = WorldContainer<Key,Node>;

inline 
void pack(int k, const Tensor<double>& s0, const Tensor<double>& s1, Tensor<double>& s) {
  assert(s0.size() == k);
  assert(s1.size() == k);
  assert(s.size() == 2*k);
  for (auto i = 0; i < k; i++) {
    s[i]   = s0[i];
    s[i+k] = s1[i];
  }
}

inline
void unpack(int k, const Tensor<double>& s, Tensor<double>& s0, Tensor<double>& s1) {
  assert(s0.size() == k);
  assert(s1.size() == k);
  assert(s.size() == 2*k);
  for (auto i = 0; i < k; i++) {
    s0[i] = s[i];
    s1[i] = s[i+k];
  }
}

class Function1D {
private:
  enum class FunctionForm {RECONSTRUCTED, COMPRESSED, UNDEFINED};

  World& world;
  FunctionForm form;
  bool debug = false;
  int k = 8;
  double thresh = 1e-6;
  int maxlevel = 30;
  int initiallevel = 4;
  CoeffTree stree;
  CoeffTree dtree;
  Tensor<double> quad_x;
  Tensor<double> quad_w;
  int quad_npts;
  Tensor<double> hg;
  Tensor<double> hgT;
  Tensor<double> quad_phi;
  Tensor<double> quad_phiT;
  Tensor<double> quad_phiw;
  Tensor<double> quad_phiwT;
  Tensor<double> r0;
  Tensor<double> rp;
  Tensor<double> rm;

public:
  // I know that friends are evil, but I decided to have them anyway
  friend Function1D operator*(const double& s, const Function1D& f);
  friend Function1D operator*(const Function1D& f, const double& s);
  friend Function1D compress(World& world, const Function1D& f);
  friend Function1D reconstruct(World& world, const Function1D& f);
  friend Function1D norm2(const Function1D& f); 

  Function1D(World& world, int k, double thresh, int maxlevel = 30, int initiallevel = 4) 
   : world(world), k(k), thresh(thresh), maxlevel(maxlevel), initiallevel(initiallevel) {
    form = FunctionForm::UNDEFINED;
    stree = CoeffTree(world);
    dtree = CoeffTree(world);
    init_twoscale(k);
    init_quadrature(k);
    make_dc_periodic(k);
  }

  Function1D(World& world, double (*f) (double), int k, double thresh, int maxlevel = 30, int initiallevel = 4) 
   : world(world), k(k), thresh(thresh), maxlevel(maxlevel), initiallevel(initiallevel) {
    form = FunctionForm::RECONSTRUCTED;
    stree = CoeffTree(world);
    dtree = CoeffTree(world);
    init_twoscale(k);
    init_quadrature(k);
    make_dc_periodic(k);
    int ntrans = std::pow(2, initiallevel);
    for (auto l = 0; l < ntrans; l++) refine(f, Key(initiallevel, l));
    for (auto n = 0; n <= initiallevel; n++) {
      ntrans = std::pow(2,n);
      for (auto l = 0; l < ntrans; l++) {
        stree.replace(Key(n,l), Node(Key(n,l), Tensor<double>()));
      }
    }
  }

  ~Function1D() {}

  bool is_compressed() const {return form == FunctionForm::COMPRESSED;}
  bool is_reconstructed() const {return form == FunctionForm::RECONSTRUCTED;} 

  void init_twoscale(int k) {
    hg = TwoScaleCoeffs::instance()->hg(k); 
    hgT = transpose(hg);
  }

  void init_quadrature(int order) {
    quad_x = gauss_legendre_x(order);
    quad_w = gauss_legendre_w(order);
    quad_npts = quad_w.size();

    quad_phi   = zeros(quad_npts, k);
    quad_phiT  = zeros(k, quad_npts);
    quad_phiw  = zeros(quad_npts, k);
    quad_phiwT = zeros(k, quad_npts);
    for (auto i = 0; i < quad_npts; i++) {
      auto p = ScalingFunction::instance()->phi(quad_x[i], k);
      for (auto m = 0; m < k; m++) {
        quad_phi(i,m) = p[m];
        quad_phiT(m,i) = p[m];
        quad_phiw(i,m) = quad_w[i]*p[m];
      }
    }
    quad_phiwT = transpose(quad_phiw); 
  }

  Tensor<double> project_box(double (*f)(double), const Key& key) {
    auto n = key.n;
    auto l = key.l;
    auto s = zeros(k);
    auto h = std::pow(0.5,n);
    auto scale = std::sqrt(h);
    for (auto mu = 0; mu < quad_npts; mu++) {
      auto x = (l + quad_x[mu]) * h;  
      auto fx = f(x);
      for (auto i = 0; i < k; i++) {
        s[i] += scale*fx*quad_phiw(mu,i);
      }
    }
    return s;
  }

  void refine(double (*f)(double), const Key& key) {
    const auto keyl = key.left_child(); 
    const auto keyr = key.right_child(); 
    auto sl = project_box(f, keyl);
    auto sr = project_box(f, keyr);
    auto s = zeros(2*k);
    pack(k, sl, sr, s);
    auto d = inner(hg,s);
    stree.replace(key, Node(key, Tensor<double>()));
    if ((d(Slice(k,2*k-1))).normf() < thresh || key.n >= maxlevel-1) {
      stree.replace(keyl, Node(keyl, sl));
      stree.replace(keyr, Node(keyr, sr));
    } else {
      refine(f, key.left_child());
      refine(f, key.right_child());
    }
  }

  double operator()(double x) {
    return eval(x, 0, 0);
  }

  double eval(double x, int n, int l) {
    assert(n < maxlevel);
    auto treep = stree.find(Key(n,l)).get();
    if (treep != stree.end() && (treep->second.coeffs).size()) {
      auto p = ScalingFunction::instance()->phi(x, k);
      auto t = wst_inner(treep->second.coeffs,p)*std::sqrt(std::pow(2.0,n));
      return t;
    } else {
      auto n2 = n + 1;
      auto l2 = 2*l;
      auto x2 = 2*x;
      if (x2 >= 1.0) {
        l2 = l2 + 1;
        x2 = x2 - 1;
      }
      return eval(x2, n2, l2);
    }
  }

  Function1D operator+(const Function1D& f) const {
    // Make sure that everybody is compressed
    assert(form == FunctionForm::COMPRESSED);
    assert(f.form == FunctionForm::COMPRESSED);
    Function1D r(world, f.k, f.thresh, f.maxlevel, f.initiallevel);
    r.form = f.form;
    auto& dtree_r = r.dtree;
    auto& stree_r = r.stree;
    // Loop over d-coeffs in this tree and add these coeffs
    // to the d-coeffs in the f tree IF THEY EXIST
    // then insert into the result
    for (auto c : dtree) {
      auto key = c.first;
      auto dcoeffs = copy(c.second.coeffs);
      auto c2 = f.dtree.find(key).get();
      if (c2 != f.dtree.end()) {
        dcoeffs += c2->second.coeffs;
        dtree_r.replace(key, Node(key, dcoeffs));
      } else {
        dtree_r.replace(key, Node(key, dcoeffs));
      }
    }
    // Loop over the remainder d-coeffs in the f tree and insert
    // into the result tree
    for (auto c : f.dtree) {
      auto key = c.first;
      auto c2 = dtree_r.find(key).get();
      if (c2 == dtree_r.end()) {
        dtree_r.replace(key,Node(key,copy(c.second.coeffs)));
      }
    }
    // Do s0 coeffs
    auto c1 = stree.find(Key(0,0)).get();
    auto c2 = f.stree.find(Key(0,0)).get();
    assert(c1 != stree.end());
    assert(c2 != f.stree.end());
    stree_r.replace(Key(0,0), Node(Key(0,0), c1->second.coeffs + c2->second.coeffs));
    return r;
  }

  Tensor<double> compress_spawn(CoeffTree& dtree_r, int n, int l) const {
    auto s0p = stree.find(Key(n+1,2*l)).get();
    auto s1p = stree.find(Key(n+1,2*l+1)).get();
    Tensor<double> s0;
    Tensor<double> s1;
    if (s0p != stree.end()) {
      s0 = ((s0p->second.coeffs).size() == 0) ? compress_spawn(dtree_r, n+1, 2*l) : s0p->second.coeffs;
    } else {
      s0 = compress_spawn(dtree_r, n+1, 2*l);
    }
    if (s1p != stree.end()) {
      s1 = ((s1p->second.coeffs).size() == 0) ? compress_spawn(dtree_r, n+1, 2*l+1) : s1p->second.coeffs;
    } else {
      s1 = compress_spawn(dtree_r, n+1, 2*l+1);
    }
    Tensor<double> s(k*2L);
    pack(k, s0, s1, s);
    Tensor<double> d = inner(hg,s);
    auto sr = d(Slice(0,k-1));
    auto dr = d(Slice(k,2*k-1));
    auto key = Key(n,l);
    dtree_r.replace(key, Node(key, dr));
    return sr;
  }

  void reconstruct_spawn(CoeffTree& stree_r, const Tensor<double>& ss, int n, int l) const {
    auto dp = dtree.find(Key(n,l)).get();
    if (dp != dtree.end()) {
      Tensor<double> dd = dp->second.coeffs;
      Tensor<double> d(k*2L);
      pack(k, ss, dd, d);
      auto s = inner(hgT,d);
      Tensor<double> s0(k);
      Tensor<double> s1(k);
      for (auto i = 0; i < k; i++) {
        s0[i] = s[i];
        s1[i] = s[i+k];
      }
      reconstruct_spawn(stree_r, s0, n+1, 2*l);
      reconstruct_spawn(stree_r, s1, n+1, 2*l+1);
      stree_r.replace(Key(n,l),Node(Key(n,l), Tensor<double>()));
    } else {
      stree_r.replace(Key(n,l),Node(Key(n,l), ss));
    }
  }

  void mul_helper(CoeffTree& r, const CoeffTree& f, const CoeffTree& g, 
                  const Tensor<double>& fsin, const Tensor<double>& gsin, int n, int l) {
    auto mrefine = true;
    auto fs = fsin;
    if (fs.size() == 0) {
      const auto fp = f.find(Key(n,l)).get();
      assert(fp != f.end());
      fs = fp->second.coeffs;
    }
    auto gs = gsin;
    if (gs.size() == 0) {
      const auto gp = g.find(Key(n,l)).get();
      assert(gp != g.end());
      gs = gp->second.coeffs;
    }
    if (fs.size() && gs.size()) {
      if (mrefine) {
        // refine to lower level for both f and g 
        Tensor<double> fd(k*2L);
        Tensor<double> gd(k*2L);
        // pack the coeffs together so that we can do two scale
        pack(k, fs, Tensor<double>(k*1L), fd);
        pack(k, gs, Tensor<double>(k*1L), gd);
        auto fss = inner(hgT,fd); auto gss = inner(hgT,gd); 
        Tensor<double> fs0(k*1L); Tensor<double> fs1(k*1L);
        Tensor<double> gs0(k*1L); Tensor<double> gs1(k*1L);
        // unpack the coeffs on n+1 level
        unpack(k, fss, fs0, fs1);
        unpack(k, gss, gs0, gs1);
        // convert to function values
        auto scale = std::sqrt(std::pow(2.0, n+1));
        auto fs0vals = inner(quad_phi, fs0);
        auto fs1vals = inner(quad_phi, fs1);
        auto gs0vals = inner(quad_phi, gs0);
        auto gs1vals = inner(quad_phi, gs1);
        auto rs0 = fs0vals.emul(gs0vals);
        auto rs1 = fs1vals.emul(gs1vals);
        rs0 = inner(quad_phiwT, rs0);
        rs1 = inner(quad_phiwT, rs1);
        rs0 = rs0.scale(scale);
        rs1 = rs1.scale(scale);
        r.replace(Key(n,l),Node(Key(n,l),Tensor<double>()));
        r.replace(Key(n+1,2*l),Node(Key(n+1,2*l),rs0));
        r.replace(Key(n+1,2*l+1),Node(Key(n+1,2*l+1),rs1));
      } else {
        // do multiply --- DOESN'T WORK!!! (me dumb)
      }
    } else {
      r.replace(Key(n,l),Node(Key(n,l),Tensor<double>()));
      mul_helper(r, f, g, fs, gs, n+1, 2*l);
      mul_helper(r, f, g, fs, gs, n+1, 2*l+1);
    }
  }

  Function1D operator*(const Function1D& g) const {
    Function1D r(world, g.k, g.thresh, g.maxlevel, g.initiallevel);
    assert(is_reconstructed());
    assert(g.is_reconstructed());
    r.mul_helper(r.stree, stree, g.stree, Tensor<double>(), Tensor<double>(), 0, 0); 
    r.form = Function1D::FunctionForm::RECONSTRUCTED;
    return r;
  }

  void make_dc_periodic(int k) {
    r0 = zeros(k,k);
    rp = zeros(k,k);
    rm = zeros(k,k);

    auto iphase = 1.0;
    for (auto i = 0; i < k; i++) {
      auto jphase = 1.0;
      for (auto j = 0; j < k; j++) {
        auto gammaij = std::sqrt((2*i+1)*(2*j+1)); 
        auto Kij = ((i-j) > 0 && (i-j) % 2== 1) ? 2.0 : 0.0;
        r0(i,j) = 0.5*(1.0 - iphase*jphase - 2.0*Kij)*gammaij;
        rm(i,j) = 0.5*jphase*gammaij;
        rp(i,j) =-0.5*iphase*gammaij;
        jphase = -jphase;
      }
      iphase = -iphase;    
    }
  }

//   void print_coeffs(int n, int l) {
//     printf("sum coeffs:\n");
//     auto s = stree[Key(n,l)];
//     printf("[%d, %d] (", n, l);
//     for (auto v : s) {
//       printf("%8.4f  ", v);
//     }
//     printf(")  %15.8e\n",normf(s));
//     printf("diff coeffs:\n");
//     auto d = dtree[Key(n,l)];
//     printf("[%d, %d] (", n, l);
//     for (auto v : d) {
//       printf("%8.4f  ", v);
//     }
//     printf(")  %15.8e\n",normf(d));
//   }
 
  void print_tree(bool docoeffs = false) {
    printf("sum coeffs:\n");
    for (auto c : stree) {
      auto k = c.first;
      auto s = c.second;
      if (docoeffs) {
        printf("[%d  %d]     %15.8e\n", k.n, k.l, normf(s));
        print(s.coeffs);
      } else {
        printf("[%d  %d]     %15.8e\n", k.n, k.l, normf(s));
      }
    }
    printf("diff coeffs:\n");
    for (auto c : dtree) {
      auto k = c.first;
      auto d = c.second; 
      if (docoeffs) {
        printf("[%d  %d]     %15.8e\n", k.n, k.l, normf(d));
        print(d.coeffs);
      } else {
        printf("[%d  %d]     %15.8e\n", k.n, k.l, normf(d));
      }
    }
  }
};

Function1D compress(World& world, const Function1D& f) {
  Function1D r(world, f.k, f.thresh, f.maxlevel, f.initiallevel);
  auto s0 = f.compress_spawn(r.dtree, 0, 0);
  r.stree.replace(Key(0,0), Node(Key(0,0), s0));
  r.form = Function1D::FunctionForm::COMPRESSED;
  return r;
}

Function1D reconstruct(World& world, const Function1D& f) {
  Function1D r(world, f.k, f.thresh, f.maxlevel, f.initiallevel);
  const auto s0 = (f.stree.find(Key(0,0)).get())->second.coeffs;
  f.reconstruct_spawn(r.stree, s0, 0, 0);
  r.form = Function1D::FunctionForm::RECONSTRUCTED;
  return r;
}
