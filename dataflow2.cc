#define WORLD_INSTANTIATE_STATIC_TEMPLATES
#include <madness/world/MADworld.h>
#include <madness/world/worlddc.h>

using namespace madness;
using namespace std;

const double L = 10.0; // The computational domain is [-L,L]
const double thresh = 1e-4; // The threshold for small difference coefficients

// Computes powers of 2
double pow2(double n) {
    return std::pow(2.0,n);
}

// Tests to see if future is assigned
template <typename T>
struct Prober {
    const Future<T>& b;
    Prober(const Future<T>& b) : b(b) {}
    bool operator()() const {return b.probe();}
};

// Waits for future to be assigned
template <typename T>
void Await(const Future<T>& t) {
    ThreadPool::await(Prober<T>(t), true);
}


// 1 dimensional index into the tree (n=level,l=translation)
struct Key {
    int n; // leave this as signed otherwise -n does unexpected things
    unsigned long l;

    Key() : n(0), l(0) {}

    Key(unsigned long n, unsigned long l) : n(n), l(l) {}

    hashT hash() const {
        hashT h = hash_value(n);
        hash_combine(h,l);
        return h;
    }

    template <typename Archive>
    void serialize(const Archive& ar) {
        ar & n & l;
    }

    bool operator==(const Key& b) const {
        return n==b.n && l==b.l;
    }

    bool operator!=(const Key& b) const {
        return !((*this)==b);
    }

    Key parent() const {
        MADNESS_ASSERT(n>0);
        return Key(n-1,l>>1); 
    }

    // Multidimensional code would provide iterator over children
    
    Key left_child() const {
        return Key(n+1,2*l);
    }
    
    Key right_child() const {
        return Key(n+1,2*l+1);
    }
};

ostream& operator<<(ostream&s, const Key& key) {
    s << "Key(" << key.n << "," << key.l << ")";
    return s;
}

// Maps middle of the box labelled by key in [0,1] to real value in [-L,L]
double key_to_x(const Key& key) {
    const double scale = (2.0*L)*pow2(-key.n);
    return -L + scale*(0.5+key.l);
}

// A node in the tree
struct Node {
    Key key; // A convenience and in multidim code is only a tiny storage overhead
    double s;
    double d;
    bool has_children;

    Node() : key(), s(0.0), d(0.0), has_children(false) {}

    Node(const Key& key, double s, double d, bool has_children) : key(key), s(s), d(d), has_children(has_children) {}

    bool operator==(const Node& a) const {
        return (key==a.key) && (std::abs(s-a.s)<1e-12) && (std::abs(d-a.d)<1e-12);

    }

    bool operator!=(const Node& a) const {
        return !(*this == a);
    }

    template <typename Archive>
    void serialize(const Archive& ar) {
        ar & key & s & d & has_children;
    }
};

ostream& operator<<(ostream&s, const Node& node) {
    s << "Node(" << node.key << "," << node.s << "," << node.d << "," << node.has_children << ")";
    return s;
}

// Pretty prints a tree by having process 0 sqeuentially traverse the entire tree left to right, depth first
void print_tree(const WorldContainer<Key,Node>& c, const Key key = Key(0,0)) {
    WorldContainer<Key,Node>::const_iterator it = c.find(key).get(); // get() "forces" the future
    if (it == c.end()) {
        error("this should not happen now we are tracking has_children");
    }
        
    for (int i=0; i<key.n; i++) std::cout << "  ";
    std::cout << it->first << " : " << it->second << " --> " << c.owner(key) << std::endl;
    if (it->second.has_children) {
        print_tree(c,key.left_child());
        print_tree(c,key.right_child());
    }
}

// Add utility functor for reduction operation
template <typename T>
struct Add {
    T operator()(const T& a, const T& b) const {return a+b;};
};
    

// And utility functor for reduction operation
template <typename T>
struct And {
    T operator()(const T& a, const T&b) const {return a && b;};
};
    

// Used to sum a value up the tree
template <typename T, typename opT>
T reduction_tree(const T& left, const T& right) {
    opT op;
    return op(left,right);
}

// Adaptively projects a function into the Haar (piecewise constant basis) storing the function values in node.s
Future<bool> project(WorldContainer<Key,Node>& c, double (*f)(double), const Key& key) {
    World& world = c.get_world();
    const double sl = f(key_to_x(key.left_child()));
    const double sr = f(key_to_x(key.right_child()));

    const double s = 0.5*(sl+sr);
    const double d = 0.5*(sl-sr);
    const double err = std::abs(d)*pow2(-0.5*key.n);
    //print(key,s,d,err);
    if ( (key.n >= 3) && (err <= thresh) ) {
        c.replace(key, Node(key,s,0.0,false));
        return Future<bool>(true);
    }
    else {
        c.replace(key, Node(key,0.0,0.0,true));
        Future <bool> left  = world.taskq.add(c.owner(key.left_child()),  project, c, f, key.left_child());
        Future <bool> right = world.taskq.add(c.owner(key.right_child()), project, c, f, key.right_child());
        return world.taskq.add(reduction_tree<bool,And<bool> >, left, right);
    }
}

Future<bool> compare(const WorldContainer<Key,Node>& a, const WorldContainer<Key,Node>& b, const Key key = Key(0,0)) {
    WorldContainer<Key,Node>::const_iterator ita = a.find(key).get();
    WorldContainer<Key,Node>::const_iterator itb = b.find(key).get();
    const bool aexists = (ita != a.end());
    const bool bexists = (itb != b.end());
    if (aexists != bexists) {
        print("compare : existence failure", key, aexists, bexists);
        return Future<bool>(false);
    }
    
    if (aexists) {
        const Node& nodea = ita->second;
        const Node& nodeb = itb->second;
        
        if (nodea != nodeb) {
            print("compare : comparison failure", key, nodea, nodeb);
            return Future<bool>(false);
        }
        
        if (nodea.has_children) {
            World& world = a.get_world();
            Future <bool> left = world.taskq.add(a.owner(key.left_child()), compare, a, b, key.left_child());
            Future <bool> right = world.taskq.add(a.owner(key.right_child()), compare, a, b, key.right_child());
            return world.taskq.add(reduction_tree<bool,And<bool> >, left, right);
        }
        else {
            return Future<bool>(true);
        }
            
    }
    else {
        print("compare : bad has_children in parent?", key);
        return Future<bool>(true);
    }        
}

double compress_op(WorldContainer<Key,Node>& c, 
                   const Key& key, 
                   const double sl,
                   const double sr)
{
    double s = 0.5*(sl+sr);
    double d = 0.5*(sl-sr);
    c.replace(key,  Node(key,(key.n==0) ? s : 0.0, d,true));
    return s;
}

Future<double> compress_spawn(WorldContainer<Key,Node>& c, const WorldContainer<Key,Node>& r, const Key& key)
{
    WorldContainer<Key,Node>::const_iterator it = r.find(key).get(); // get() "forces" the future
    const Node& node = it->second;
    if (node.has_children) {
        World& world = c.get_world();
        Key k_left = key.left_child();
        Key k_right = key.right_child();
        Future <double> f_left =  world.taskq.add(c.owner(k_left), compress_spawn, c, r, k_left);
        Future <double> f_right = world.taskq.add(c.owner(k_right), compress_spawn, c, r, k_right);
        return world.taskq.add(c.owner(key), compress_op, c, key, f_left, f_right);
    }
    else {
        c.replace(key, Node(key,0.0,0.0,false)); // insert empty leaf at bottom of compressed tree
        return Future<double>(node.s);
    }
    
}

Future<bool> reconstruct(WorldContainer<Key,Node>& r, const WorldContainer<Key,Node>& c, const Key& key, double s = 0.0) {
    World& world = c.get_world();
    WorldContainer<Key,Node>::const_iterator it = c.find(key).get(); // get() "forces" the future
    const Node& node = it->second;

    if (key == Key(0,0)) s = node.s;

    if (node.has_children) {
        r.replace(key, Node(key,0.0,0.0,true));
        double d = node.d;
        double sl = (s+d);
        double sr = (s-d);
        Key k_left = key.left_child();
        Key k_right = key.right_child();
        Future<bool> f_left = world.taskq.add(c.owner(k_left), reconstruct, r, c, k_left, sl);
        Future<bool> f_right = world.taskq.add(c.owner(k_right), reconstruct, r, c, k_right, sr);
        return world.taskq.add(reduction_tree<bool,And<bool> >, f_left, f_right);
    }
    else {
        r.replace(key, Node(key,s,0.0,false));
        return Future<bool>(true);
    }
}

// Computes the trace (int(f(x),x=-L,L)) of the function represented by the tree
Future<double> trace(const WorldContainer<Key,Node>& c, const Key& key) {
    World& world = c.get_world();
    WorldContainer<Key,Node>::const_iterator it = c.find(key).get(); // get() "forces" the future
    const Node& node = it->second;

    if (node.has_children) {
        Future <double> left  = world.taskq.add(c.owner(key.left_child()),  trace, c, key.left_child());
        Future <double> right = world.taskq.add(c.owner(key.right_child()), trace, c, key.right_child());
        return world.taskq.add(reduction_tree<double,Add<double> >, left, right);
    }
    else {
        return Future<double>(node.s*pow2(-key.n)*L); // The factor is the box size
    }
}    
    
double func(const double x) {
    return std::exp(-x*x);
}

void test0(World& world) {
    Key root(0,0);
    WorldContainer<Key,Node> c(world);
    WorldContainer<Key,Node> f1(world);
    WorldContainer<Key,Node> f2(world);
    if (world.rank() == 0) {
        Await(world.taskq.add(c.owner(root), project, c, &func, root));
        print_tree(c);
        Future<double> sum = world.taskq.add(c.owner(root), trace, c, root);
        Await(sum);
        print("sum",sum,"exact",sqrt(3.1415926535/4.0));
        Await(world.taskq.add(c.owner(root), compress_spawn, f1, c, root)); // compress c into f1
        print("compressed");
        Await(world.taskq.add(f1.owner(root), reconstruct, f2, f1, root, 0.0)); // reconstruct f1 into f2
        print("reconstructed");
        Future<bool> test = world.taskq.add(c.owner(root), compare, c, f2, Key(0,0));
        Await(test);
        print("comparison", test.get());
        f2.replace(Key(1,0),Node(Key(1,0),99.0,99.0,1)); // Inject an error to check compare
        Future<bool> test2 = world.taskq.add(c.owner(root), compare, c, f2, Key(0,0));
        Await(test2);
        print("comparison2", test2.get());
    }

    c.get_world().gop.fence(); // Only here so that other processes don't run too far ahead ... should turn into massively threaded pool
}

int main(int argc, char** argv) {
    initialize(argc, argv);
    World world(SafeMPI::COMM_WORLD);
    try {
        test0(world);
    }
    catch (SafeMPI::Exception e) {
        error("caught an MPI exception");
    }
    catch (madness::MadnessException e) {
        print("XXX",e);
        error("caught a MADNESS exception");
    }
    catch (const char* s) {
        print(s);
        error("caught a string exception");
    }
    catch (...) {
        error("caught unhandled exception");
    }

    finalize();
    return 0;
}
