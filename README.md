
## Model Architecture

| Layer      | Units | Activation |
|------------|-------|------------|
| Input      | *12*   | -          |
| Linear 1   | 64    | PReLU      |
| Linear 2   | 32    | PReLU      |
| Output     | 1     | -          |

- **Loss:** Mean Squared Error (MSE)
- **Optimizer:** SGD (lr=0.001)
- **Batch size:** 64
- **Epochs:** 100
 ---
  Test loss: 0.50
