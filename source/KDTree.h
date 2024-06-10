#ifndef KDTREE_H__
#define KDTREE_H__

#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>

/**
 * @file KDTree.h
 * @brief Declaration of KDTree class.
 */

/**
 * @brief Node structure for KDTree.
 */
struct KDNode
{
    std::vector<double> point; /**< Data point coordinates. */
    int label; /**< Label of the data point. */
    KDNode* left; /**< Pointer to the left child node. */
    KDNode* right; /**< Pointer to the right child node. */

    /**
     * @brief Constructor for KDNode.
     * @param pt Data point coordinates.
     * @param lbl Label of the data point.
     */
    KDNode(const std::vector<double>& pt, int lbl) : point(pt), label(lbl), left(nullptr), right(nullptr) {}
};

/**
 * @brief Implementation of KDTree for nearest neighbor search.
 */
class KDTree
{
public:
    KDTree();
    ~KDTree();

    /**
     * @brief Build the KDTree from given data points.
     * @param points Data points.
     * @param labels Labels corresponding to the data points.
     */
    void build(const std::vector<std::vector<double>>& points, const std::vector<int>& labels);

    /**
     * @brief Find the nearest neighbor to a given point.
     * @param point Query point.
     * @return Label of the nearest neighbor.
     */
    int nearestNeighbor(const std::vector<double>& point);

    /**
     * @brief Get the closest distances to a given point from k nearest neighbors.
     * @param point Query point.
     * @param k Number of nearest neighbors.
     * @return Closest distances to the query point.
     */
    std::vector<double> getClosestDistances(const std::vector<double>& point, int k);

private:
    KDNode* root; /**< Pointer to the root node of the KDTree. */
    int dimension; /**< Dimensionality of the data points. */

    /**
     * @brief Recursively build the KDTree.
     * @param points Data points.
     * @param depth Depth of the current node in the tree.
     * @return Root node of the KDTree.
     */
    KDNode* buildTree(std::vector<std::pair<std::vector<double>, int>>& points, int depth);

    /**
     * @brief Destroy the KDTree recursively.
     * @param node Root node of the subtree to be destroyed.
     */
    void destroyTree(KDNode* node);

    /**
     * @brief Find the nearest neighbor to a given point recursively.
     * @param root Root node of the subtree.
     * @param target Query point.
     * @param best Nearest neighbor found so far.
     * @param best_dist Distance to the nearest neighbor found so far.
     * @param depth Depth of the current node in the tree.
     */
    void nearest(KDNode* root, const std::vector<double>& target, KDNode*& best, double& best_dist, int depth) const;

    /**
     * @brief Calculate the Euclidean distance between two points.
     * @param a First point.
     * @param b Second point.
     * @return Euclidean distance between the points.
     */
    double calculateDistance(const std::vector<double>& a, const std::vector<double>& b) const;

    /**
     * @brief Find k nearest neighbors to a given point.
     * @param root Root node of the subtree.
     * @param target Query point.
     * @param closest_distances Priority queue to store the closest distances.
     * @param k Number of nearest neighbors.
     * @param depth Depth of the current node in the tree.
     */
    void findKNearest(KDNode* root, const std::vector<double>& target, std::priority_queue<double>& closest_distances, int k, int depth);
};

#endif // KDTREE_H__
