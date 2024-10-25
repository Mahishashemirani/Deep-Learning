# Optimal Rotation Age Prediction using Neural ODE🌲 

## 📖 Introduction

Neural Ordinary Differential Equations (Neural ODEs) are a powerful tool for modeling continuous-time dynamical systems. This project demonstrates how Neural ODEs can be utilized to predict the optimal rotation age for timber growth, focusing on maximizing the Net Present Value (NPV) from timber sales. By capturing the underlying dynamics of timber growth through time, this model provides valuable insights for forestry management and decision-making.

## 🔄 Workflow

1. **Data Simulation**  
   - Generate synthetic data representing timber growth over time using a specified growth function.
   - Calculate the Net Present Value (NPV) for various rotation ages to determine the optimal age for harvesting.

2. **Neural ODE Model Architecture**  
   - **Input**: The initial timber volume and a time vector.
   - **ODE Function**: A neural network that models the growth dynamics based on historical data.
   - **Output**: Predictions of timber growth over time.

3. **Training**  
   - **Loss Function**: Mean Squared Error (MSE) to minimize the prediction error.
   - **Optimizer**: Adam optimizer for efficient training.
   - **Validation**: Split data into training and validation sets to evaluate model performance.

4. **Prediction and Evaluation**  
   - Use the trained Neural ODE model to predict timber growth over time.
   - Evaluate model performance using MSE.
   - Visualize predicted vs. actual growth to interpret the model's performance.

---

## 📉 Loss Function in Neural ODE for Timber Growth Prediction

The **loss function** plays a crucial role in training the Neural ODE model, quantifying the discrepancy between predicted and actual timber volumes. For this project, we employ the **Mean Squared Error (MSE)** loss function, which calculates the average squared difference between the predicted and actual values. MSE is particularly suitable for regression tasks like timber growth prediction, as it heavily penalizes larger errors, prompting the model to minimize significant deviations.

During the training process, the Neural ODE model aims to minimize the MSE by adjusting its weights through backpropagation using the **Adam optimizer**. Observing the trend of the loss function over epochs provides insights into the model’s learning journey. A **decreasing loss** signifies improved predictions, while a **static or oscillating loss** may indicate convergence issues or hyperparameter tuning needs.

Below is the plot showing the loss function over the course of training:

![Training Loss Plot](plots/training%20loss.png)

## 🏋️ Training Procedure of Neural ODE and Observed Patterns

Training a **Neural ODE for timber growth prediction** involves feeding sequential data into the model, adjusting its internal states, and updating weights via backpropagation. Throughout each epoch, the model processes batches of time windows to predict future growth based on historical data. The **Adam optimizer** facilitates effective learning by minimizing the loss function. Initially, predictions may seem erratic as the model struggles to grasp the underlying patterns.

As training continues, several patterns may be observed:
1. **Gradual Improvement**: Predictions begin to align more closely with actual growth values as the model learns.
2. **Overfitting**: The model may excel on the training set but falter on unseen data, indicating the need for regularization.
3. **Training Instability**: Fluctuations in predictions may occur during training, which could necessitate adjustments to learning rate or network architecture.
4. **Convergence**: The model achieves stable and accurate predictions, demonstrating effective learning and generalization from the data.

Below is a plot comparing the **predicted growth vs actual growth** over time, offering a visual representation of the model's performance:

![Predicted vs Actual Growth](plots/prediction%20vs.%20true.png)

## 📊 Gradient Norms Over Training Epochs

The gradient norms during training provide insights into the stability and efficiency of the learning process. Below is a plot showing the gradient norms over the training epochs, indicating how the model's learning dynamics evolved:

![Gradient Norms Plot](plots/gradient%20norms.png)

## 📈 Visualization of Predictions

This visualization helps assess the model's ability to track trends in timber growth over time, illustrating the predictions against the actual values in the dataset.

![Predicted vs Actual Prices](plots/visualization.png)

## 🔚 Conclusion

This project highlights the application of Neural ODEs in forecasting timber growth and determining optimal rotation ages. The insights gained from this model can significantly aid in forestry management, optimizing harvest times for economic sustainability.


