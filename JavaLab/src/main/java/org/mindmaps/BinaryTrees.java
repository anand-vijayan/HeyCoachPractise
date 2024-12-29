package org.mindmaps;

import org.dto.Node;
import org.dto.TreeNode;

import java.util.*;

public class BinaryTrees {
    //region Variables & Constants
    private static int totalTilt = 0;
    private static int sum = 0;
    private static int[][] dist;
    private static Map<Integer, List<Integer>> graph;
    private static int[] count;
    private static int[] res;
    private static int resource = 0;
    private static int maxSum = 0;
    private static int maximumSum = Integer.MIN_VALUE;
    //endregion

    //region Binary Tree
    public static boolean RootEqualsSumOfChildren(TreeNode root) {
        return root.val == (root.left.val + root.right.val);
    }

    public static TreeNode IncreasingOrderSearchTree_1(TreeNode root) {
        return Stacks.IncreasingOrderSearchTree(root);
    }

    public static List<Integer> BinaryTreePostorderTraversal(TreeNode root){
        return Stacks.BinaryTreePostOrderTraversal(root);
    }

    public static List<Integer> BinaryTreePreorderTraversal(TreeNode root){
        return Stacks.BinaryTreePreOrderTraversal(root);
    }

    public static int BinaryTreeTilt(TreeNode root) {
        calculateSum(root);
        return totalTilt;
    }
    //endregion

    //region Binary Search Tree
    public static int RangeSumOfBST_1(TreeNode root, int low, int high) {
        return dfs(root, low, high);
    }

    public static TreeNode SearchInABinarySearchTree(TreeNode root, int val) {
        if (root == null || root.val == val) {
            return root;  // Either found the node or reached a null node
        }

        // If val is less than root's value, search in the left subtree
        if (val < root.val) {
            return SearchInABinarySearchTree(root.left, val);
        }

        // If val is greater than root's value, search in the right subtree
        return SearchInABinarySearchTree(root.right, val);
    }

    public static List<Integer> BinaryTreeInorderTraversal(TreeNode root){
        return Stacks.BinaryTreeInorderTraversal(root);
    }

    public static TreeNode ConvertSortedArrayToBinarySearchTree_1(int[] nums) {
        return DivideAndConquer.ConvertSortedArrayToBinarySearchTree(nums);
    }

    public TreeNode BinarySearchTreeToGreaterSumTree(TreeNode root) {
        reverseInOrder(root);
        return root;
    }
    //endregion

    //region DFS Based
    public static int RangeSumOfBST_2(TreeNode root, int low, int high) {
        return RangeSumOfBST_1(root, low, high);
    }

    public static TreeNode IncreasingOrderSearchTree_2(TreeNode root){
        return IncreasingOrderSearchTree_1(root);
    }

    public static TreeNode MergeTwoBinaryTrees_1(TreeNode root1, TreeNode root2) {
        // If both nodes are null, return null
        if (root1 == null && root2 == null) {
            return null;
        }

        // If one of the nodes is null, return the other node
        if (root1 == null) {
            return root2;
        }
        if (root2 == null) {
            return root1;
        }

        // If both nodes are non-null, sum their values and recursively merge left and right subtrees
        root1.val += root2.val;

        // Recursively merge left and right children
        root1.left = MergeTwoBinaryTrees_1(root1.left, root2.left);
        root1.right = MergeTwoBinaryTrees_1(root1.right, root2.right);

        return root1;
    }

    public static int SumOfRootToLeafBinaryNumbers(TreeNode root) {
        return dfs(root, 0);
    }

    public static int MaximumDepthOfBinaryTree_1(TreeNode root) {
        if (root == null) {
            return 0;  // If the tree is empty, depth is 0
        }

        // Recursively find the max depth of the left and right subtrees
        int leftDepth = MaximumDepthOfBinaryTree_1(root.left);
        int rightDepth = MaximumDepthOfBinaryTree_1(root.right);

        // The depth of the current node is 1 + the maximum of the left and right subtree depths
        return Math.max(leftDepth, rightDepth) + 1;
    }
    //endregion

    //region BFS Based
    public static TreeNode MergeTwoBinaryTrees_2(TreeNode root1, TreeNode root2) {
        return MergeTwoBinaryTrees_1(root1, root2);
    }

    public static int MaximumDepthOfBinaryTree_2(TreeNode root){
        return MaximumDepthOfBinaryTree_1(root);
    }

    public static void InvertBinaryTree(TreeNode root) {
        if (root == null) {
            return;  // If the tree is empty, return null
        }

        // Swap the left and right children of the current node
        TreeNode temp = root.left;
        root.left = root.right;
        root.right = temp;

        // Recursively invert the left and right subtrees
        InvertBinaryTree(root.left);
        InvertBinaryTree(root.right);

    }

    public static boolean UniValuedBinaryTree(TreeNode root) {
        if (root == null) {
            return true;  // An empty tree is considered uni-valued
        }

        // Call the helper function to check if all nodes have the same value
        return isUniValuedBTHelper(root, root.val);
    }

    public static List<Double> AverageOfLevelsInBinaryTree(TreeNode root) {
        List<Double> averages = new ArrayList<>();

        if (root == null) {
            return averages;
        }

        // Queue for level-order traversal (BFS)
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);

        while (!queue.isEmpty()) {
            int levelSize = queue.size();
            double levelSum = 0;

            // Process all nodes at the current level
            for (int i = 0; i < levelSize; i++) {
                TreeNode node = queue.poll();
                levelSum += node.val;

                // Add children to the queue for the next level
                if (node.left != null) {
                    queue.offer(node.left);
                }
                if (node.right != null) {
                    queue.offer(node.right);
                }
            }

            // Calculate the average for this level and add it to the result list
            averages.add(levelSum / levelSize);
        }

        return averages;
    }
    //endregion

    //region Divide And Conquer
    public static TreeNode MaximumBinaryTree(int[] nums) {
        return DivideAndConquer.MaximumBinaryTree(nums);
    }

    public static TreeNode ConvertSortedArrayToBinarySearchTree_2(int[] nums) {
        return ConvertSortedArrayToBinarySearchTree_1(nums);
    }

    public static TreeNode BalanceABinarySearchTree(TreeNode root) {
        List<Integer> nodes = new ArrayList<>();

        // Perform inorder traversal and collect node values
        inorder(root, nodes);

        // Convert the sorted array into a balanced BST
        return sortedArrayToBST(nodes, 0, nodes.size() - 1);
    }
    //endregion

    //region String Matching
    public static boolean SubtreeOfAnotherTree(TreeNode root, TreeNode subRoot) {
        if (root == null) {
            return false;
        }

        // If the current node matches, check if the whole subtree matches
        if (isSameTree(root, subRoot)) {
            return true;
        }

        // Recursively check the left and right subtrees of the current node
        return SubtreeOfAnotherTree(root.left, subRoot) || SubtreeOfAnotherTree(root.right, subRoot);
    }
    //endregion

    //region N-Ary Tree
    public static List<Integer> NaryTreePreOrderTraversal(Node root) {
        return Stacks.NaryTreePreOrderTraversal(root);
    }

    public static List<Integer> NaryTreePostOrderTraversal(Node root) {
        return Stacks.NaryTreePostOrderTraversal(root);
    }

    public static int MaximumDepthOfNaryTree(Node root) {
        if (root == null) {
            return 0;  // Base case: if the node is null, the depth is 0.
        }

        int maxChildDepth = 0;
        // Recursively calculate the depth of all children.
        for (Node child : root.children) {
            maxChildDepth = Math.max(maxChildDepth, MaximumDepthOfNaryTree(child));
        }

        // The depth of the current node is 1 plus the maximum depth of its children.
        return maxChildDepth + 1;
    }
    //endregion

    //region Backtracking
    public static List<String> BinaryTreePaths(TreeNode root){
        return Strings.BinaryTreePaths(root);
    }
    //endregion

    //region DP Based
    public static int[] CountSubtreesWithMaxDistanceBetweenCities(int n, int[][] edges) {
        //precalculate the distance of any two cities
        dist = new int[n][n];
        List<Integer>[] graph = new List[n];
        for(int i=0; i<n; i++){
            graph[i] = new ArrayList<>();
        }
        for(int[] e : edges){
            graph[e[0]-1].add(e[1]-1);
            graph[e[1]-1].add(e[0]-1);
        }
        for(int i=0; i<n; i++){
            calcDist(graph, i, -1, i, 0);
        }
        int[] res = new int[n-1];
        for(int i=1; i<(1<<n); i++){
            int maxDist = 0;
            int oneDistCount = 0;
            int cities = 0;
            //find the max distance between each pair of cities
            for(int j=0; j<n; j++){
                if((i & (1<<j))!=0){
                    cities++;
                    for(int k=j+1; k<n; k++){
                        if((i & (1<<k))!=0){
                            maxDist = Math.max(maxDist, dist[j][k]);
                            if(dist[j][k]==1){
                                oneDistCount++;
                            }
                        }
                    }
                }
            }
            //x cities form a substree if and only if there are x-1 edges among these cities
            if(oneDistCount==cities-1 && maxDist>0) res[maxDist-1]++;
        }
        return res;
    }

    public static int[] SumOfDistancesInTree(int n, int[][] edges) {
        graph = new HashMap<>();
        count = new int[n];
        res = new int[n];
        Arrays.fill(count, 1);

        for (int i = 0; i < n; i++) {
            graph.put(i, new ArrayList<>());
        }

        for (int[] edge : edges) {
            int u = edge[0];
            int v = edge[1];
            graph.get(u).add(v);
            graph.get(v).add(u);
        }

        dfs(0, -1);
        dfs2(0, -1);

        return res;
    }

    public static int BinaryTreeCameras(TreeNode root) {
        return (dfs(root) < 1 ? 1 : 0) + resource;
    }

    public int MaximumSumBSTInBinaryTree(TreeNode root) {
        postOrderTraverse(root);
        return maxSum;
    }

    public int BinaryTreeMaximumPathSum(TreeNode root) {
        helper(root);
        return maximumSum;
    }

    //endregion

    //region Private Methods
    private static int calculateSum(TreeNode node) {
        if (node == null) {
            return 0;
        }

        // Recursively get the sum of left and right subtrees
        int leftSum = calculateSum(node.left);
        int rightSum = calculateSum(node.right);

        // Calculate the tilt for the current node
        int tilt = Math.abs(leftSum - rightSum);
        totalTilt += tilt;  // Add the tilt of this node to the total tilt

        // Return the sum of the current subtree
        return node.val + leftSum + rightSum;
    }

    private static int dfs(TreeNode node, int low, int high) {
        if (node == null) {
            return 0;  // Base case: if the node is null, return 0
        }

        int sum = 0;

        // If the current node's value is within the range, include it in the sum
        if (node.val >= low && node.val <= high) {
            sum += node.val;
        }

        // Recursively check the left subtree if the current node's value is greater than low
        if (node.val > low) {
            sum += dfs(node.left, low, high);
        }

        // Recursively check the right subtree if the current node's value is less than high
        if (node.val < high) {
            sum += dfs(node.right, low, high);
        }

        return sum;
    }

    private static void reverseInOrder(TreeNode node) {
        if (node == null) {
            return;
        }

        // Traverse the right subtree first (greater values)
        reverseInOrder(node.right);

        // Update the current node's value
        sum += node.val;  // Add the current node's value to the sum
        node.val = sum;   // Set the node's value to the accumulated sum

        // Traverse the left subtree
        reverseInOrder(node.left);
    }

    private static int dfs(TreeNode node, int currentValue) {
        if (node == null) {
            return 0;
        }

        // Update the current binary number at this node
        currentValue = (currentValue << 1) | node.val;

        // If it's a leaf node, return the current binary number
        if (node.left == null && node.right == null) {
            return currentValue;
        }

        // Otherwise, recursively calculate the sum for both subtrees
        return dfs(node.left, currentValue) + dfs(node.right, currentValue);
    }

    private static boolean isUniValuedBTHelper(TreeNode node, int value) {
        if (node == null) {
            return true;  // Null nodes are fine, they don't violate the uni-valued condition
        }

        // If the current node's value does not match the expected value, return false
        if (node.val != value) {
            return false;
        }

        // Recursively check the left and right subtrees
        return isUniValuedBTHelper(node.left, value) && isUniValuedBTHelper(node.right, value);
    }

    private static void inorder(TreeNode root, List<Integer> nodes) {
        if (root == null) return;
        inorder(root.left, nodes);
        nodes.add(root.val);  // Add the node's value to the list
        inorder(root.right, nodes);
    }

    private static TreeNode sortedArrayToBST(List<Integer> nodes, int left, int right) {
        if (left > right) return null;

        // Find the middle element to make it the root
        int mid = left + (right - left) / 2;
        TreeNode root = new TreeNode(nodes.get(mid));

        // Recursively construct the left and right subtrees
        root.left = sortedArrayToBST(nodes, left, mid - 1);
        root.right = sortedArrayToBST(nodes, mid + 1, right);

        return root;
    }

    private static boolean isSameTree(TreeNode tree1, TreeNode tree2) {
        // If both are null, they are the same
        if (tree1 == null && tree2 == null) {
            return true;
        }

        // If one is null and the other is not, they are not the same
        if (tree1 == null || tree2 == null) {
            return false;
        }

        // Check if the current nodes' values are the same and recursively check left and right subtrees
        return (tree1.val == tree2.val) && isSameTree(tree1.left, tree2.left) && isSameTree(tree1.right, tree2.right);
    }

    public static void calcDist(List<Integer>[] graph, int source, int prev, int cur, int d){
        if(prev==cur){
            return;
        }
        dist[source][cur] = d;
        dist[cur][source] = d;
        for(int next : graph[cur]) if(next!=prev){
            calcDist(graph, source, cur, next, d+1);
        }
    }

    private static void dfs(int node, int parent) {
        for (int child : graph.get(node)) {
            if (child != parent) {
                dfs(child, node);
                count[node] += count[child];
                res[node] += res[child] + count[child];
            }
        }
    }

    private static void dfs2(int node, int parent) {
        for (int child : graph.get(node)) {
            if (child != parent) {
                res[child] = res[node] - count[child] + (count.length - count[child]);
                dfs2(child, node);
            }
        }
    }

    private static int dfs(TreeNode root) {
        if (root == null) return 2;
        int left = dfs(root.left), right = dfs(root.right);
        if (left == 0 || right == 0) {
            resource++;
            return 1;
        }
        return left == 1 || right == 1 ? 2 : 0;
    }

    private static int[] postOrderTraverse(TreeNode root) {
        if (root == null) return new int[]{Integer.MAX_VALUE, Integer.MIN_VALUE, 0}; // {min, max, sum}, initialize min=MAX_VALUE, max=MIN_VALUE
        int[] left = postOrderTraverse(root.left);
        int[] right = postOrderTraverse(root.right);
        // The BST is the tree:
        if (!(     left != null             // the left subtree must be BST
                && right != null            // the right subtree must be BST
                && root.val > left[1]       // the root's key must greater than maximum keys of the left subtree
                && root.val < right[0]))    // the root's key must lower than minimum keys of the right subtree
            return null;
        int sum = root.val + left[2] + right[2]; // now it's a BST make `root` as root
        maxSum = Math.max(maxSum, sum);
        int min = Math.min(root.val, left[0]);
        int max = Math.max(root.val, right[1]);
        return new int[]{min, max, sum};
    }

    private static int helper(TreeNode node) {
        if (node == null) {
            return 0;
        }

        int leftMaxPath = Math.max(helper(node.left), 0);
        int rightMaxPath = Math.max(helper(node.right), 0);

        int maxIfNodeIsRoot = node.val + leftMaxPath + rightMaxPath;
        maximumSum = Math.max(maximumSum, maxIfNodeIsRoot);

        return node.val + Math.max(leftMaxPath, rightMaxPath);
    }
    //endregion
}
