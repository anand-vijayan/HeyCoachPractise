package org.helpers;

import java.util.LinkedList;
import java.util.Queue;

public class RecentCounter {

    private Queue<Integer> queue;

    public RecentCounter() {
        queue = new LinkedList<>();
    }

    public int ping(int t) {
        queue.add(t);
        // Remove timestamps that are not in the range [t - 3000, t]
        while (!queue.isEmpty() && queue.peek() < t - 3000) {
            queue.poll();
        }
        // The size of the queue is the number of requests in the range
        return queue.size();
    }
}
