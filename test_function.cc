#define WORLD_INSTANTIATE_STATIC_TEMPLATES
#include "function1d.h"

const double PI = 3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825;
const int k = 8;
const double thresh = 1e-4;
const int initiallevel = 1;

double linear(double x) {
  return x;
}

double quadratic(double x) {
  return x*x;
}

double func1(double x) {
  auto c = 1.0;
  auto mu = 2.0;
  return c*sin(2*PI*x)*std::exp(-mu*x);
}

double dfunc1(double x) {
  auto c = 1.0;
  auto mu = 2.0;
  return c*cos(2*PI*x)*std::exp(-mu*x)-mu*c*sin(2*PI*x)*std::exp(-mu*x);
}

double func2(double x) {
  auto c = 1.0;
  auto mu = 2.0;
  return c*cos(2*PI*x)*std::exp(-mu*x);
}

double sum_f12(double x) {
  return func1(x) + func2(x);
}

double mul_f12(double x) {
  return func1(x) * func2(x);
}

void test_function_add(int argc, char** argv) {
  initialize(argc, argv);
  World world(SafeMPI::COMM_WORLD);
  auto fr = Function1D(world, func1, k, thresh, 30, initiallevel);
  auto fc = compress(world, fr);
  printf("\nfunction fc:\n");
  fr.print_tree();
  auto gr = Function1D(world, func2, k, thresh, 30, initiallevel);
  auto gc = compress(world, gr);
  printf("\nfunction gc:\n");
  gr.print_tree();
  auto fgc = fc + gc;
  printf("\nfunction f+g(c):\n");
  fgc.print_tree();
  auto f12r = Function1D(world, sum_f12, k, thresh, 30, initiallevel);
  auto f12c = compress(world, f12r);
  printf("\nfunction f12c:\n");
  f12c.print_tree();
  finalize();
}
void test_function_compress(int argc, char** argv) {
  initialize(argc, argv);
  World world(SafeMPI::COMM_WORLD);
  Function1D f(world, func1, k, thresh, 30, initiallevel);
  printf("f : \n");
  f.print_tree();
  Function1D g = reconstruct(world, compress(world, f));
  printf("\ng : \n");
  g.print_tree();
  finalize();
}

void test_function_mul(int argc, char** argv) {
  initialize(argc, argv);
  World world(SafeMPI::COMM_WORLD);
  auto fr = Function1D(world, func1, k, thresh, 30, initiallevel);
  printf("\nfunction fr:\n");
  fr.print_tree();
  auto gr = Function1D(world, func2, k, thresh, 30, initiallevel);
  printf("\nfunction gr:\n");
  gr.print_tree();
  auto fgr = fr * gr;
  printf("\nfunction f+g(r):\n");
  fgr.print_tree();
  auto f12r = Function1D(world, mul_f12, k, thresh, 30, initiallevel+1);
  printf("\nfunction f12r:\n");
  f12r.print_tree();

  auto x = 0.23111;
  printf("x: %15.8e f: %15.8e func: %15.8e error: %15.8e\n", x, fgr(x), f12r(x), std::abs(fgr(x)-f12r(x)));
  finalize();
}
void test_function_point(int argc, char** argv) {
  initialize(argc, argv);
  World world(SafeMPI::COMM_WORLD);
  Function1D f(world, func1, k, thresh, 30, initiallevel);
  f.print_tree();
  auto x = 0.23111;
  printf("x: %15.8e f: %15.8e func: %15.8e error: %15.8e\n", x, f(x), func1(x), std::abs(f(x)-func1(x)));
  finalize();
}


int main(int argc, char** argv) {
  // test_function_point(argc,argv);
  test_function_compress(argc,argv);
  // test_function_add(argc,argv);
  // test_function_mul(argc,argv);
  return 0;
}
