package org.mindmaps;

import org.helpers.*;

import java.util.*;

public class Queues {
    //region Design
    public static void ImplementStackUsingQueues() {
        MyStack stack = new MyStack();

        // Test case
        stack.push(1);
        stack.push(2);
        System.out.println(stack.top());  // Output: 2
        System.out.println(stack.pop());  // Output: 2
        System.out.println(stack.empty()); // Output: false
        System.out.println(stack.pop());  // Output: 1
        System.out.println(stack.empty()); // Output: true
    }

    public static void ImplementQueueUsingStacks(){
        MyQueue queue = new MyQueue();

        // Test case
        queue.push(1);
        queue.push(2);
        System.out.println(queue.peek()); // Output: 1
        System.out.println(queue.pop());  // Output: 1
        System.out.println(queue.empty()); // Output: false
        System.out.println(queue.pop());  // Output: 2
        System.out.println(queue.empty()); // Output: true
    }

    public static void NumberOfRecentCalls() {
        RecentCounter recentCounter = new RecentCounter();
        System.out.println(recentCounter.ping(1));     // Output: 1
        System.out.println(recentCounter.ping(100));   // Output: 2
        System.out.println(recentCounter.ping(3001));  // Output: 3
        System.out.println(recentCounter.ping(3002));  // Output: 3
    }

    public static void DesignCircularQueue() {
        MyCircularQueue circularQueue = new MyCircularQueue(3); // Set the size to 3
        System.out.println(circularQueue.enQueue(1)); // Output: true
        System.out.println(circularQueue.enQueue(2)); // Output: true
        System.out.println(circularQueue.enQueue(3)); // Output: true
        System.out.println(circularQueue.enQueue(4)); // Output: false (queue is full)
        System.out.println(circularQueue.Rear());     // Output: 3
        System.out.println(circularQueue.isFull());   // Output: true
        System.out.println(circularQueue.deQueue());  // Output: true
        System.out.println(circularQueue.enQueue(4)); // Output: true
        System.out.println(circularQueue.Rear());     // Output: 4
    }

    public static void DesignCircularDeque(){
        MyCircularDeque circularDeque = new MyCircularDeque(3); // Set the size to 3
        System.out.println(circularDeque.insertLast(1)); // Output: true
        System.out.println(circularDeque.insertLast(2)); // Output: true
        System.out.println(circularDeque.insertFront(3)); // Output: true
        System.out.println(circularDeque.insertFront(4)); // Output: false (deque is full)
        System.out.println(circularDeque.getRear());      // Output: 2
        System.out.println(circularDeque.isFull());       // Output: true
        System.out.println(circularDeque.deleteLast());   // Output: true
        System.out.println(circularDeque.insertFront(4)); // Output: true
        System.out.println(circularDeque.getFront());     // Output: 4
    }
    //endregion

    //region Simulation
    public static int TimeNeededToBuyATicket(int[] tickets, int k) {
        int time = 0;
        int n = tickets.length;

        // Loop until the person at position k finishes buying all their tickets
        for (int i = 0; i < n; i = (i + 1) % n) {
            // Only process the person if they still need to buy tickets
            if (tickets[i] > 0) {
                tickets[i]--; // Decrease the ticket count for the current person
                time++; // Increment the total time

                // If we are at position k and they have finished buying all their tickets
                if (i == k && tickets[k] == 0) {
                    break;
                }
            }
        }

        return time;
    }

    public static int NumberOfStudentsUnableToEat(int[] students, int[] sandwiches) {
        Queue<Integer> queue = new LinkedList<>();
        for (int student : students) {
            queue.offer(student); // Add all students to the queue
        }

        int index = 0; // Index for the stack of sandwiches
        int attempts = 0; // Tracks how many consecutive students have failed to eat

        while (!queue.isEmpty() && attempts < queue.size()) {
            if (queue.peek() == sandwiches[index]) {
                // Student takes the sandwich
                queue.poll(); // Remove the student from the queue
                index++; // Move to the next sandwich
                attempts = 0; // Reset failed attempts
            } else {
                // Student does not take the sandwich
                queue.offer(queue.poll()); // Move the student to the end of the queue
                attempts++; // Increment failed attempts
            }
        }

        return queue.size();
    }

    public static int MaximumSumCircularSubArray(int[] nums) {
        int n = nums.length;

        // Calculate the total sum of the array
        int totalSum = 0;

        // Find the maximum subarray sum using Kadane's algorithm
        int maxSum = Integer.MIN_VALUE;
        int currentMax = 0;

        // Find the minimum subarray sum using Kadane's algorithm
        int minSum = Integer.MAX_VALUE;
        int currentMin = 0;

        for (int num : nums) {
            totalSum += num;

            // Kadane's for maximum subarray sum
            currentMax = Math.max(currentMax + num, num);
            maxSum = Math.max(maxSum, currentMax);

            // Kadane's for minimum subarray sum
            currentMin = Math.min(currentMin + num, num);
            minSum = Math.min(minSum, currentMin);
        }

        // If all elements are negative, return the maximum element
        if (maxSum < 0) {
            return maxSum;
        }

        // The maximum of non-circular and circular cases
        return Math.max(maxSum, totalSum - minSum);
    }

    public static int FindTheWinnerOfTheCircularGame1(int n, int k) {
        ArrayList<Integer> friends = new ArrayList<>();
        for (int i = 1; i <= n; i++) {
            friends.add(i);
        }

        int index = 0; // Start from the first friend
        while (friends.size() > 1) {
            // Find the next friend to be removed
            index = (index + k - 1) % friends.size();
            friends.remove(index);
        }

        return friends.get(0); // The last remaining friend
    }

    public static int[] RevealCardsInIncreasingOrder(int[] deck) {
        // Step 1: Sort the deck
        Arrays.sort(deck);

        // Step 2: Use a deque to simulate the reverse process
        Deque<Integer> deque = new LinkedList<>();
        for (int i = deck.length - 1; i >= 0; i--) {
            if (!deque.isEmpty()) {
                deque.addFirst(deque.removeLast());
            }
            deque.addFirst(deck[i]);
        }

        // Step 3: Convert the deque to an array and return
        int[] result = new int[deck.length];
        int index = 0;
        for (int card : deque) {
            result[index++] = card;
        }
        return result;
    }
    //endregion

    //region Recursion
    public static int FindTheWinnerOfTheCircularGame2(int n, int k){
        return FindTheWinnerOfTheCircularGame1(n,k);
    }
    //endregion

    //region Priority Queue Based
    public static int JumpGame1(int[] nums, int k) {
        int n = nums.length;
        int[] dp = new int[n];
        dp[0] = nums[0];

        Deque<Integer> deque = new LinkedList<>();
        deque.add(0); // Initialize deque with the first index

        for (int i = 1; i < n; i++) {
            // Remove indices outside the sliding window
            if (!deque.isEmpty() && deque.peekFirst() < i - k) {
                deque.pollFirst();
            }

            // The maximum dp value for the current range is at the front of the deque
            dp[i] = nums[i] + dp[deque.peekFirst()];

            // Maintain deque in decreasing order of dp values
            while (!deque.isEmpty() && dp[i] >= dp[deque.peekLast()]) {
                deque.pollLast();
            }

            // Add the current index to the deque
            deque.addLast(i);
        }

        // The maximum score to reach the last index
        return dp[n - 1];
    }

    public static int MaxValueOfEquation1(int[][] points, int k) {
        // Deque to maintain indices of points
        Deque<int[]> deque = new LinkedList<>();
        int maxVal = Integer.MIN_VALUE;

        for (int[] point : points) {
            int xj = point[0], yj = point[1];

            // Remove points from the deque where |xj - xi| > k
            while (!deque.isEmpty() && xj - deque.peekFirst()[0] > k) {
                deque.pollFirst();
            }

            // Update max value using the front of the deque
            if (!deque.isEmpty()) {
                maxVal = Math.max(maxVal, yj + xj + deque.peekFirst()[1]);
            }

            // Compute vj = yj - xj
            int vj = yj - xj;

            // Maintain the deque in decreasing order of v values
            while (!deque.isEmpty() && deque.peekLast()[1] <= vj) {
                deque.pollLast();
            }

            // Add the current point to the deque
            deque.addLast(new int[]{xj, vj});
        }

        return maxVal;
    }

    public static int LongestContinuousSubArrayWithAbsoluteDiffLessThanOrEqualToLimit1(int[] nums, int limit) {
        // Deques to maintain the max and min values in the current window
        Deque<Integer> maxDeque = new LinkedList<>();
        Deque<Integer> minDeque = new LinkedList<>();

        int left = 0;
        int maxLength = 0;

        // Traverse the array with the right pointer (j)
        for (int right = 0; right < nums.length; right++) {
            // Maintain the max deque (non-increasing order)
            while (!maxDeque.isEmpty() && nums[maxDeque.peekLast()] <= nums[right]) {
                maxDeque.pollLast();
            }
            maxDeque.offerLast(right);

            // Maintain the min deque (non-decreasing order)
            while (!minDeque.isEmpty() && nums[minDeque.peekLast()] >= nums[right]) {
                minDeque.pollLast();
            }
            minDeque.offerLast(right);

            // Shrink the window if the absolute difference between max and min exceeds limit
            while (nums[maxDeque.peekFirst()] - nums[minDeque.peekFirst()] > limit) {
                left++;

                // Remove elements that are out of the window from deques
                if (maxDeque.peekFirst() < left) {
                    maxDeque.pollFirst();
                }
                if (minDeque.peekFirst() < left) {
                    minDeque.pollFirst();
                }
            }

            // Update the maxLength with the size of the current valid window
            maxLength = Math.max(maxLength, right - left + 1);
        }

        return maxLength;
    }
    //endregion

    //region Sliding Window
    public static int JumpGame2(int[] numbers, int k) {
        return JumpGame1(numbers,k);
    }

    public static int MaxValueOfEquation2(int[][] points, int k) {
        return MaxValueOfEquation1(points,k);
    }

    public static int ShortestSubArrayWithSumAtLeastK(int[] A, int K) {
        int N = A.length, res = N + 1;
        long[] B = new long[N + 1];
        for (int i = 0; i < N; i++) B[i + 1] = B[i] + A[i];
        Deque<Integer> d = new ArrayDeque<>();
        for (int i = 0; i < N + 1; i++) {
            while (!d.isEmpty() && B[i] - B[d.getFirst()] >=  K)
                res = Math.min(res, i - d.pollFirst());
            while (!d.isEmpty() && B[i] <= B[d.getLast()])
                d.pollLast();
            d.addLast(i);
        }
        return res <= N ? res : -1;
    }

    public static int LongestContinuousSubArrayWithAbsoluteDiffLessThanOrEqualToLimit2(int[] nums, int limit) {
        return LongestContinuousSubArrayWithAbsoluteDiffLessThanOrEqualToLimit1(nums, limit);
    }
    //endregion

    //region Hashing
    public static int FirstUniqueCharacterInAString(String s) {
        HashMap<Character, Integer> freqMap = new HashMap<>();

        // Count the frequency of each character
        for (char c : s.toCharArray()) {
            freqMap.put(c, freqMap.getOrDefault(c, 0) + 1);
        }

        // Find the first character with a frequency of 1
        for (int i = 0; i < s.length(); i++) {
            if (freqMap.get(s.charAt(i)) == 1) {
                return i;
            }
        }

        // If no non-repeating character is found, return -1
        return -1;
    }
    //endregion
}
