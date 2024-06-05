#ifndef KDTREE_H__
#define KDTREE_H__

#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>

struct KDNode
{
    std::vector<int> point;
    int label;
    KDNode* left;
    KDNode* right;

    KDNode(const std::vector<int>& pt, int lbl) : point(pt), label(lbl), left(nullptr), right(nullptr) {}
};

class KDTree
{
public:
    KDTree();
    ~KDTree();

    void build(const std::vector<std::vector<int>>& points, const std::vector<int>& labels);
    int nearestNeighbor(const std::vector<int>& point);
    std::vector<double> getClosestDistances(const std::vector<int>& point, int k);

private:
    KDNode* root;
    int dimension;

    KDNode* buildTree(std::vector<std::pair<std::vector<int>, int>>& points, int depth);
    void destroyTree(KDNode* node);
    void nearest(KDNode* root, const std::vector<int>& target, KDNode*& best, double& best_dist, int depth) const;
    double calculateDistance(const std::vector<int>& a, const std::vector<int>& b) const;
    void findKNearest(KDNode* root, const std::vector<int>& target, std::priority_queue<double>& closest_distances, int k, int depth);
};

#endif // KDTREE_H__
