import java.awt.*;
import java.awt.event.*;
import javax.swing.*;

public class GameOfLife extends JFrame implements ActionListener {

    private static final int ROWS = 50;
    private static final int COLUMNS = 50;
    private static final int CELL_SIZE = 10;
    private static final int DELAY = 100;

    private boolean[][] grid;
    private boolean[][] newGrid;
    private JButton startButton, stopButton, clearButton;
    private JPanel gridPanel;
    private Timer timer;

    public GameOfLife() {
        setTitle("Game of Life");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLayout(new BorderLayout());

        // Create buttons
        startButton = new JButton("Start");
        startButton.addActionListener(this);
        stopButton = new JButton("Stop");
        stopButton.addActionListener(this);
        clearButton = new JButton("Clear");
        clearButton.addActionListener(this);

        // Create button panel
        JPanel buttonPanel = new JPanel();
        buttonPanel.add(startButton);
        buttonPanel.add(stopButton);
        buttonPanel.add(clearButton);
        add(buttonPanel, BorderLayout.NORTH);

        // Create grid panel
        gridPanel = new JPanel();
        gridPanel.setLayout(new GridLayout(ROWS, COLUMNS));
        gridPanel.setPreferredSize(new Dimension(COLUMNS * CELL_SIZE, ROWS * CELL_SIZE));
        add(gridPanel, BorderLayout.CENTER);

        // Create grid
        grid = new boolean[ROWS][COLUMNS];
        newGrid = new boolean[ROWS][COLUMNS];
        for (int i = 0; i < ROWS; i++) {
            for (int j = 0; j < COLUMNS; j++) {
                grid[i][j] = Math.random() < 0.5;
                JPanel cellPanel = new JPanel();
                cellPanel.setBackground(grid[i][j] ? Color.BLACK : Color.WHITE);
                gridPanel.add(cellPanel);
            }
        }

        // Create timer
        timer = new Timer(DELAY, this);
    }

    public void actionPerformed(ActionEvent e) {
        if (e.getSource() == startButton) {
            timer.start();
        } else if (e.getSource() == stopButton) {
            timer.stop();
        } else if (e.getSource() == clearButton) {
            clearGrid();
        } else if (e.getSource() == timer) {
            updateGrid();
        }
    }

    private void updateGrid() {
        // Update new grid based on old grid
        for (int i = 0; i < ROWS; i++) {
            for (int j = 0; j < COLUMNS; j++) {
                int neighbors = countNeighbors(i, j);
                if (grid[i][j]) {
                    newGrid[i][j] = neighbors == 2 || neighbors == 3;
                } else {
                    newGrid[i][j] = neighbors == 3;
                }
            }
        }

        // Update grid and display on screen
        for (int i = 0; i < ROWS; i++) {
            for (int j = 0; j < COLUMNS; j++) {
                grid[i][j] = newGrid[i][j];
                JPanel cellPanel = (JPanel) gridPanel.getComponent(i * COLUMNS + j);
                cellPanel.setBackground(grid[i][j] ? Color.BLACK : Color.WHITE);
            }
        }
        gridPanel.repaint();
    }

    private int countNeighbors(int row, int col) {
        int count = 0;
        for (int i = row - 1; i <= row + 1; i++) {
            for (
