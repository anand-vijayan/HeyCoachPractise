package org.dto;

public class Line implements Comparable<Line> {
    public int length;
    public int p1;
    public int p2;

    public Line(int p1, int p2, int length) {
        this.p1 = p1;
        this.p2 = p2;
        this.length = length;
    }

    @Override
    public int compareTo(Line other) {
        return Integer.compare(this.length, other.length);
    }
}
