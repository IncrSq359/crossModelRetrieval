/*
HNSW的cpp实现版本，
functions:
    Insert(Tensor: nq)
    Search(Tensor: nq, int K, int efSearch = -1)
*/

#include <algorithm>
#include <cmath>
#include <deque>
#include <numeric>
#include <random>
#include <vector>
using namespace std;

struct Node;

// 堆，c=0为小根堆，c=1为大根堆
struct Heap {
    int lc, flag, sz = 0, c, dim;
    vector<int> heap;
    vector<float> distan;
    float* q;
    vector<vector<Node>>* g;

    template <typename T>
    bool cmp(T x, T y) {
        return c == 0 ? x < y : x > y;
    }
    Heap(int lc, int flag, int c, int dim, float* q, vector<vector<Node>>* g) {
        this->lc = lc;
        this->flag = flag;
        this->c = c;
        this->q = q;
        this->dim = dim;
        this->g = g;
    }
    // 需要读 Node::data，定义放在 Node 完整定义之后
    void push(int idx, float d = -1);
    pair<int, float> top() {
        if (sz == 0)
            return {-1, -1};
        return {heap[0], distan[0]};
    }
    void pop() {
        if (sz == 0)
            return;
        swap(heap[0], heap[sz - 1]);
        swap(distan[0], distan[sz - 1]);
        heap.pop_back();
        distan.pop_back();
        sz--;
        sortDown();
    }
    void sortUp(int idx) {
        int child = idx;
        int fa = (child - 1) / 2;
        while (fa >= 0) {
            if (cmp(distan[child], distan[fa])) {
                swap(heap[child], heap[fa]);
                swap(distan[child], distan[fa]);
                child = fa;
                fa = (child - 1) / 2;
            } else
                break;
        }
    }
    void sortDown() {
        int fa = 0, child = fa * 2 + 1;
        while (child < sz) {
            if (child < sz - 1 && cmp(distan[child + 1], distan[child])) {
                child++;
            }
            if (cmp(distan[child], distan[fa])) {
                swap(heap[fa], heap[child]);
                swap(distan[fa], distan[child]);
                fa = child;
                child = 2 * fa + 1;
            } else
                break;
        }
    }
};

struct Node {
    int id;
    float* data;
    Heap edges;  // 小根堆
    int next;
};

// flag: 0采用欧氏距离，1采用余弦距离
float dis(const float* x, const float* q, int len, int flag) {
    if (flag == 0) {
        float res = 0;
        for (int i = 0; i < len; i++) {
            res += (x[i] - q[i]) * (x[i] - q[i]);
        }
        return res;
    } else {
        float dot = inner_product(x, x + len, q, 0.0f);
        float A = inner_product(x, x + len, x, 0.0f);
        float B = inner_product(q, q + len, q, 0.0f);
        return (A == 0.f || B == 0.f) ? 1.f : 1.f - dot / (sqrtf(A) * sqrtf(B));
    }
}

inline void Heap::push(int idx, float d) {
    heap.push_back(idx);
    distan.push_back(d != -1 ? d : dis((*g)[lc][idx].data, q, this->dim, flag));
    ++sz;
    sortUp(sz - 1);
}
class HNSWindex {
   public:
    static uniform_real_distribution<float> dist;
    HNSWindex(int dim, int m, int efConstruction, int efSearch, int flag = 0) {
        this->efConstruction = efConstruction;
        this->efSearch = efSearch;
        this->dim = dim;
        this->m = m;
        this->ml = 1 / log(m);
        this->flag = flag;
        random_device sd;
        this->generator = mt19937(sd());
    }
    // 生成指数概率随机数：P(level >= k) = m^(-k)
    int RandomLayer() {
        float u = dist(this->generator);
        return floor(-log(1.0f - u) * this->ml);
    }

    void Insert(float* nq, size_t n) {
        for (size_t k = 0; k < n; k++) {
            // 拷贝一份由索引自己持有：调用方的缓冲区（尤其是 pybind11 的临时
            // array）在 Insert 返回后随时可能释放，只存裸指针会读到野内存
            float* q = this->Store(nq + k * this->dim);
            vector<int> ep = {this->ep};
            int L = topLayer;
            int l = RandomLayer();
            for (int lc = L; lc > l; lc--) {
                Heap W = SearchLayer(q, ep, 1, lc);
                ep[0] = W.heap[0];
                ep[0] = hnsw[lc][ep[0]].next;
            }

            int nid = tolidx++;
            if (l > L) {
                for (int i = L + 1; i <= l; i++) {
                    hnsw.emplace_back(vector<Node>());
                    Node nodeq = {
                        nid, q, Heap(i, this->flag, 0, this->dim, q, &hnsw), 0};
                    hnsw[i].emplace_back(nodeq);
                    if (i != L + 1)
                        hnsw[i][0].next = 0;
                }
                this->ep = 0;
                this->topLayer = l;
            }
            int last = -1;
            for (int lc = min(L, l); lc >= 0; lc--) {
                Node nodeq = {nid, q,
                              Heap(lc, this->flag, 0, this->dim, q, &hnsw), 0};
                hnsw[lc].emplace_back(nodeq);
                int idx = hnsw[lc].size() - 1;
                if (l > L && lc == L) {
                    hnsw[L + 1][0].next = idx;
                } else if (last != -1) {
                    hnsw[lc + 1][last].next = idx;
                }
                last = idx;
                Heap W = SearchLayer(q, ep, this->efConstruction, lc);
                int Mmax = (lc == 0) ? 2 * this->m : this->m;
                Heap neighbors = SelectNeighborsHeuristic(q, W, Mmax, lc, 0, 1);
                for (int i = 0; i < neighbors.sz; i++) {
                    float disn = neighbors.distan[i];
                    int idxn = neighbors.heap[i];
                    hnsw[lc][idxn].edges.push(idx, disn);
                    hnsw[lc][idx].edges.push(idxn, disn);
                }
                for (int i = 0; i < neighbors.sz; i++) {
                    Heap& C = hnsw[lc][neighbors.heap[i]].edges;
                    if (C.sz > Mmax) {
                        int idxn = neighbors.heap[i];
                        hnsw[lc][neighbors.heap[i]].edges =
                            SelectNeighborsHeuristic(hnsw[lc][idxn].data, C,
                                                     Mmax, lc, 0, 0);
                    }
                }
                ep.clear();
                for (int idx : W.heap) {
                    ep.push_back(hnsw[lc][idx].next);
                }
            }
        }
    }

    void KNNSearch(float* nq, int n, int K,float* D,int* I ,int ef = -1) {
        if (hnsw.empty())
            return ;  // 空索引
        float* q = nq;
        ef = ef == -1 ? this->efSearch : ef;
        for (int j = 0; j < n; j++, q += this->dim) {
            vector<int> ep = {this->ep};
            int L = topLayer;
            for (int i = L; i > 0; i--) {
                Heap W = SearchLayer(q, ep, 1, i);
                ep[0] = W.heap[0];
                ep[0] = hnsw[i][ep[0]].next;
            }
            Heap W = SearchLayer(q, ep, max(ef, K), 0);
            W.c = 0;
            for (int i = 0; i < W.sz; i++)
                W.sortUp(i);
            int kk = min(K, W.sz);
            for (int i = 0; i < kk; i++) {
                I[j * K + i] = hnsw[0][W.heap[0]].id;
                D[j * K + i] = W.distan[0];
                W.pop();
            }
        }
    }
    // 返回大根堆
    Heap SearchLayer(float* q, vector<int> ep, int ef, int lc) {
        Heap candidates(lc, flag, 0, this->dim, q, &hnsw);  // 小根堆
        Heap W(lc, flag, 1, this->dim, q, &hnsw);           // 大根堆
        vector<char> visited(hnsw[lc].size(), 0);
        for (int idx : ep) {
            candidates.push(idx);
            W.push(idx);
            visited[idx] = 1;
        }
        while (candidates.sz > 0) {
            auto c = candidates.top();
            candidates.pop();
            auto f = W.top();
            if (f.first != -1 && c.second > f.second)
                break;
            Heap& edges = hnsw[lc][c.first].edges;
            for (int i = 0; i < edges.sz; i++) {
                if (visited[edges.heap[i]] == 0) {
                    visited[edges.heap[i]] = 1;
                    float dise =
                        dis(hnsw[lc][edges.heap[i]].data, q, this->dim, flag);
                    if (W.sz < ef || dise < f.second) {
                        candidates.push(edges.heap[i], dise);
                        W.push(edges.heap[i], dise);
                        if (W.sz > ef)
                            W.pop();
                    }
                }
            }
        }
        return W;
    }
    Heap SelectNeighborsHeuristic(float* q,
                                  Heap& C,
                                  int M,
                                  int lc,
                                  int exC,
                                  int kPC) {
        Heap W = C;
        if (C.c == 1) {
            W.c = 0;
            for (int i = 0; i < W.sz; i++)
                W.sortUp(i);
        }
        Heap R(lc, this->flag, 0, this->dim, q, &hnsw);
        if (exC) {
            for (int idx : C.heap) {
                Heap& neighbors = hnsw[lc][idx].edges;
                for (int i = 0; i < neighbors.sz; i++) {
                    int k = 1;
                    for (int j : W.heap)
                        if (neighbors.heap[i] == j) {
                            k = 0;
                            break;
                        }
                    if (k)
                        W.push(neighbors.heap[i]);
                }
            }
        }
        Heap Wd(lc, this->flag, 0, this->dim, q, &hnsw);
        while (W.sz > 0 && R.sz < M) {
            auto e = W.top();
            W.pop();
            int k = 1;
            float dise = e.second;
            for (int i = 0; i < R.sz; i++) {
                float disr =
                    dis(hnsw[lc][e.first].data, hnsw[lc][R.heap[i]].data,
                        this->dim, this->flag);
                if (disr < dise) {
                    k = 0;
                    break;
                }
            }
            if (k) {
                R.push(e.first, e.second);
            } else
                Wd.push(e.first, e.second);
        }
        if (kPC) {
            while (Wd.sz > 0 && R.sz < M) {
                auto e = Wd.top();
                R.push(e.first, e.second);
                Wd.pop();
            }
        }
        return R;
    }

   private:
    // 每个点一份独立的 vector：地址稳定（deque 尾插不移动已有元素），
    // 而且每份数据自身连续，Node::data 才能按 dim 个 float 读。
    // （注意不能用 deque<float>：MSVC 的 deque 每块只有 4 个 float，不连续）
    float* Store(const float* src) {
        data_store.emplace_back(src, src + this->dim);
        return data_store.back().data();
    }

    int efConstruction, efSearch;
    int dim, m;
    int tolidx = 0;
    int topLayer = -1;
    int flag = 0;
    float ml;
    int ep = 0;
    deque<vector<float>> data_store;
    vector<vector<Node>> hnsw;  // layer,point
    mt19937 generator;
};
inline uniform_real_distribution<float> HNSWindex::dist =
    uniform_real_distribution<float>(0, 1);
