#ifndef GAUSS_LEGENDRE_H_
#define GAUSS_LEGENDRE_H_

using GLVector = std::vector<double>;

#define GL_POINTS0 = {}
#define GL_WEIGHTS0 = {}

#define GL_POINTS1 = {0.50000000000000000}
#define GL_WEIGHTS1 = { 1.00000000000000000}

#define GL_POINTS2 = {0.78867513459481287, 0.21132486540518713}
#define GL_WEIGHTS2 = { 0.50000000000000011, 0.50000000000000011}

#define GL_POINTS3 = {0.88729833462074170, 0.50000000000000000, 0.11270166537925830}
#define GL_WEIGHTS3 = { 0.27777777777777751, 0.44444444444444442, 0.27777777777777751}

#define GL_POINTS4 = {0.93056815579702623, 0.66999052179242813, 0.33000947820757187, 0.06943184420297371}
#define GL_WEIGHTS4 = { 0.17392742256872701, 0.32607257743127305, 0.32607257743127305, 0.17392742256872701}

#define GL_POINTS5 = {0.95308992296933193, 0.76923465505284150, 0.50000000000000000, 0.23076534494715845, 0.04691007703066802}
#define GL_WEIGHTS5 = { 0.11846344252809465, 0.23931433524968321, 0.28444444444444444, 0.23931433524968321, 0.11846344252809465}

#define GL_POINTS6 = {0.96623475710157603, 0.83060469323313224, 0.61930959304159849, 0.38069040695840156, 0.16939530676686776, 0.03376524289842397}
#define GL_WEIGHTS6 = { 0.08566224618958508, 0.18038078652406936, 0.23395696728634546, 0.23395696728634546, 0.18038078652406936, 0.08566224618958508}

#define GL_POINTS7 = {0.97455395617137930, 0.87076559279969723, 0.70292257568869854, 0.50000000000000000, 0.29707742431130141, 0.12923440720030277, 0.02544604382862070}
#define GL_WEIGHTS7 = { 0.06474248308443417, 0.13985269574463832, 0.19091502525255946, 0.20897959183673470, 0.19091502525255946, 0.13985269574463832, 0.06474248308443417}

#define GL_POINTS8 = {0.98014492824876809, 0.89833323870681336, 0.76276620495816450, 0.59171732124782495, 0.40828267875217511, 0.23723379504183550, 0.10166676129318664, 0.01985507175123191}
#define GL_WEIGHTS8 = { 0.05061426814518921, 0.11119051722668717, 0.15685332293894369, 0.18134189168918102, 0.18134189168918102, 0.15685332293894369, 0.11119051722668717, 0.05061426814518921}

#define GL_POINTS9 = {0.98408011975381304, 0.91801555366331788, 0.80668571635029518, 0.66212671170190451, 0.50000000000000000, 0.33787328829809554, 0.19331428364970482, 0.08198444633668206, 0.01591988024618696}
#define GL_WEIGHTS9 = { 0.04063719418078738, 0.09032408034742866, 0.13030534820146775, 0.15617353852000135, 0.16511967750062989, 0.15617353852000135, 0.13030534820146775, 0.09032408034742841, 0.04063719418078738}

#define GL_POINTS10 = {0.98695326425858587, 0.93253168334449232, 0.83970478414951222, 0.71669769706462361, 0.57443716949081558, 0.42556283050918442, 0.28330230293537639, 0.16029521585048778, 0.06746831665550773, 0.01304673574141413}
#define GL_WEIGHTS10 = { 0.03333567215434403, 0.07472567457529021, 0.10954318125799088, 0.13463335965499817, 0.14776211235737641, 0.14776211235737641, 0.13463335965499817, 0.10954318125799088, 0.07472567457529021, 0.03333567215434403}
#define GL_POINTS11 = {0.98911432907302843, 0.94353129988404771, 0.86507600278702468, 0.75954806460340585, 0.63477157797617245, 0.50000000000000000, 0.36522842202382755, 0.24045193539659410, 0.13492399721297532, 0.05646870011595234, 0.01088567092697151}
#define GL_WEIGHTS11 = { 0.02783428355808731, 0.06279018473245226, 0.09314510546386703, 0.11659688229599521, 0.13140227225512346, 0.13646254338895031, 0.13140227225512346, 0.11659688229599521, 0.09314510546386703, 0.06279018473245226, 0.02783428355808731}

Tensor<double> gauss_legendre_x(int k) {
  if (k == 0) {
    GLVector x GL_POINTS0;
    Tensor<double> tx;
    return tx;
  }
  else if (k == 1) {
    GLVector x GL_POINTS1;
    Tensor<double> tx(k*1L);
    for (auto i = 0; i < k; i++) tx[i] = x[i];
    return tx;
  }
  else if (k == 2) {
    GLVector x GL_POINTS2;
    Tensor<double> tx(k*1L);
    for (auto i = 0; i < k; i++) tx[i] = x[i];
    return tx;
  }
  else if (k == 3) {
    GLVector x GL_POINTS3;
    Tensor<double> tx(k*1L);
    for (auto i = 0; i < k; i++) tx[i] = x[i];
    return tx;
  }
  else if (k == 4) {
    GLVector x GL_POINTS4;
    Tensor<double> tx(k*1L);
    for (auto i = 0; i < k; i++) tx[i] = x[i];
    return tx;
  }
  else if (k == 5) {
    GLVector x GL_POINTS5;
    Tensor<double> tx(k*1L);
    for (auto i = 0; i < k; i++) tx[i] = x[i];
    return tx;
  }
  else if (k == 6) {
    GLVector x GL_POINTS6;
    Tensor<double> tx(k*1L);
    for (auto i = 0; i < k; i++) tx[i] = x[i];
    return tx;
  }
  else if (k == 7) {
    GLVector x GL_POINTS7;
    Tensor<double> tx(k*1L);
    for (auto i = 0; i < k; i++) tx[i] = x[i];
    return tx;
  }
  else if (k == 8) {
    GLVector x GL_POINTS8;
    Tensor<double> tx(k*1L);
    for (auto i = 0; i < k; i++) tx[i] = x[i];
    return tx;
  }
  else if (k == 9) {
    GLVector x GL_POINTS9;
    Tensor<double> tx(k*1L);
    for (auto i = 0; i < k; i++) tx[i] = x[i];
    return tx;
  }
  else if (k == 10) {
    GLVector x GL_POINTS10;
    Tensor<double> tx(k*1L);
    for (auto i = 0; i < k; i++) tx[i] = x[i];
    return tx;
  }
  else if (k == 11) {
    GLVector x GL_POINTS11;
    Tensor<double> tx(k*1L);
    for (auto i = 0; i < k; i++) tx[i] = x[i];
    return tx;
  }
  else {
    assert(false);
  }
}

Tensor<double> gauss_legendre_w(int k) {
  if (k == 0) {
    GLVector w GL_WEIGHTS0;
    Tensor<double> tw;
    return tw;
  }
  else if (k == 1) {
    GLVector w GL_WEIGHTS1;
    Tensor<double> tw(k*1L);
    for (auto i = 0; i < k; i++) tw[i] = w[i];
    return tw;
  }
  else if (k == 2) {
    GLVector w GL_WEIGHTS2;
    Tensor<double> tw(k*1L);
    for (auto i = 0; i < k; i++) tw[i] = w[i];
    return tw;
  }
  else if (k == 3) {
    GLVector w GL_WEIGHTS3;
    Tensor<double> tw(k*1L);
    for (auto i = 0; i < k; i++) tw[i] = w[i];
    return tw;
  }
  else if (k == 4) {
    GLVector w GL_WEIGHTS4;
    Tensor<double> tw(k*1L);
    for (auto i = 0; i < k; i++) tw[i] = w[i];
    return tw;
  }
  else if (k == 5) {
    GLVector w GL_WEIGHTS5;
    Tensor<double> tw(k*1L);
    for (auto i = 0; i < k; i++) tw[i] = w[i];
    return tw;
  }
  else if (k == 6) {
    GLVector w GL_WEIGHTS6;
    Tensor<double> tw(k*1L);
    for (auto i = 0; i < k; i++) tw[i] = w[i];
    return tw;
  }
  else if (k == 7) {
    GLVector w GL_WEIGHTS7;
    Tensor<double> tw(k*1L);
    for (auto i = 0; i < k; i++) tw[i] = w[i];
    return tw;
  }
  else if (k == 8) {
    GLVector w GL_WEIGHTS8;
    Tensor<double> tw(k*1L);
    for (auto i = 0; i < k; i++) tw[i] = w[i];
    return tw;
  }
  else if (k == 9) {
    GLVector w GL_WEIGHTS9;
    Tensor<double> tw(k*1L);
    for (auto i = 0; i < k; i++) tw[i] = w[i];
    return tw;
  }
  else if (k == 10) {
    GLVector w GL_WEIGHTS10;
    Tensor<double> tw(k*1L);
    for (auto i = 0; i < k; i++) tw[i] = w[i];
    return tw;
  }
  else if (k == 11) {
    GLVector w GL_WEIGHTS11;
    Tensor<double> tw(k*1L);
    for (auto i = 0; i < k; i++) tw[i] = w[i];
    return tw;
  }
  else {
    assert(false);
  }
}

#endif
