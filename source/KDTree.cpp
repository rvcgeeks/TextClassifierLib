
#include "KDTree.h"

KDTree::KDTree() : root(nullptr), dimension(0) {}

KDTree::~KDTree()
{
    destroyTree(root);
}

void KDTree::build(const std::vector<std::vector<int>>& points, const std::vector<int>& labels)
{
    if (points.empty()) return;
    dimension = points[0].size();
    std::vector<std::pair<std::vector<int>, int>> point_labels(points.size());
    for (size_t i = 0; i < points.size(); ++i)
    {
        point_labels[i] = std::make_pair(points[i], labels[i]);
    }
    root = buildTree(point_labels, 0);
}

KDNode* KDTree::buildTree(std::vector<std::pair<std::vector<int>, int>>& points, int depth)
{
    if (points.empty()) return nullptr;

    int axis = depth % dimension;
    size_t median = points.size() / 2;
    std::nth_element(points.begin(), points.begin() + median, points.end(),
                     [axis](const std::pair<std::vector<int>, int>& a, const std::pair<std::vector<int>, int>& b)
                     {
                         return a.first[axis] < b.first[axis];
                     });

    KDNode* node = new KDNode(points[median].first, points[median].second);
    std::vector<std::pair<std::vector<int>, int>> left(points.begin(), points.begin() + median);
    std::vector<std::pair<std::vector<int>, int>> right(points.begin() + median + 1, points.end());
    node->left = buildTree(left, depth + 1);
    node->right = buildTree(right, depth + 1);
    
    return node;
}

void KDTree::destroyTree(KDNode* node)
{
    if (!node) return;
    destroyTree(node->left);
    destroyTree(node->right);
    delete node;
}

int KDTree::nearestNeighbor(const std::vector<int>& point)
{
    KDNode* best = nullptr;
    double best_dist = std::numeric_limits<double>::infinity();
    nearest(root, point, best, best_dist, 0);
    return best->label;
}

void KDTree::nearest(KDNode* root, const std::vector<int>& target, KDNode*& best, double& best_dist, int depth) const
{
    if (!root) return;

    double d = calculateDistance(root->point, target);
    if (d < best_dist)
    {
        best_dist = d;
        best = root;
    }

    int axis = depth % dimension;
    KDNode* next = target[axis] < root->point[axis] ? root->left : root->right;
    KDNode* other = target[axis] < root->point[axis] ? root->right : root->left;

    nearest(next, target, best, best_dist, depth + 1);
    if (std::abs(target[axis] - root->point[axis]) < best_dist)
    {
        nearest(other, target, best, best_dist, depth + 1);
    }
}

double KDTree::calculateDistance(const std::vector<int>& a, const std::vector<int>& b) const
{
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}
