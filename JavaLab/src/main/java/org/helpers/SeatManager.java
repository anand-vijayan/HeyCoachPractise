package org.helpers;

import java.util.PriorityQueue;

public class SeatManager {
    private PriorityQueue<Integer> availableSeats;

    public SeatManager(int n) {
        // Initialize the priority queue with all seats from 1 to n
        availableSeats = new PriorityQueue<>();
        for (int i = 1; i <= n; i++) {
            availableSeats.add(i);
        }
    }

    public int reserve() {
        // Fetch and remove the smallest numbered available seat
        return availableSeats.poll();
    }

    public void unreserve(int seatNumber) {
        // Add the seat back to the priority queue (marking it as available)
        availableSeats.add(seatNumber);
    }
}
