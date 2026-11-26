# Q-learning and Neural Network Control of a Taxi Robot

6ELEN018C.1 Applied Robotics coursework — M Y Imanjith (w2052278)

A robot taxi in the Gymnasium **Taxi-v4** environment is controlled in two ways:

1. **Q-learning** implemented from scratch (Part 1)
2. **A neural network** (sklearn `MLPClassifier`) trained on the Q-learning optimum actions (Part 2)

## Files

| File | Description |
|---|---|
| `part1_q_learning_controller.ipynb` | Q-learning: baseline table, per-configuration training, evaluation, and creation of the dataset |
| `part2_neural_network_controller.ipynb` | Neural network: tuning, train/test split, metrics, and the network driving the taxi |
| `taxi_dataset_404.csv` | The 404-row dataset `(taxi_row, taxi_col, pass_idx, dest_idx) -> best_action`, created by Part 1 and read by Part 2 |
| `taxi_dataset_one_hot_encode.csv` | The one-hot encoded inputs, created by Part 2 |

## Requirements

- Python 3.11
- JupyterLab
- `gymnasium`, `numpy`, `pandas`, `matplotlib`, `scikit-learn`

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install jupyterlab gymnasium numpy pandas matplotlib scikit-learn
jupyter lab
```

## How to run

Keep all files in the same folder, then:

1. Open `part1_q_learning_controller.ipynb` and choose **Run → Restart Kernel and Run All Cells**. This creates `taxi_dataset_404.csv`.
2. Open `part2_neural_network_controller.ipynb` and choose **Run → Restart Kernel and Run All Cells**.

Part 1 must be run first, because Part 2 reads the CSV it creates.
