import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import roc_auc_score, precision_recall_curve, classification_report, confusion_matrix, average_precision_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import random
from scipy import stats
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class CAPTCHABotDetector:
    def __init__(self, random_seed=42):
        """Initialize the CAPTCHA bot detection system."""
        np.random.seed(random_seed)
        random.seed(random_seed)
        self.model = None
        self.feature_names = []
        
    def generate_human_session(self, session_id: int) -> Dict:
        """Generate human pointer movement session with specified user types."""
        human_types = ['casual_user', 'power_user', 'elderly_user', 'cad_designer']
        human_type = np.random.choice(human_types)
        
        # Configure parameters based on user type
        if human_type == 'casual_user':
            num_events = np.random.randint(25, 70)
            base_speed = np.random.uniform(0.3, 0.8)
            precision = np.random.uniform(0.7, 0.9)
            pause_probability = 0.12
            
        elif human_type == 'power_user':
            num_events = np.random.randint(15, 40)
            base_speed = np.random.uniform(0.8, 1.2)
            precision = np.random.uniform(0.8, 0.95)
            pause_probability = 0.05
            
        elif human_type == 'elderly_user':
            num_events = np.random.randint(30, 90)
            base_speed = np.random.uniform(0.2, 0.5)
            precision = np.random.uniform(0.6, 0.8)
            pause_probability = 0.25
            
        else:  # cad_designer
            num_events = np.random.randint(20, 50)
            base_speed = np.random.uniform(0.6, 0.9)
            precision = np.random.uniform(0.9, 0.98)
            pause_probability = 0.08

        # Initialize position
        x, y = np.random.randint(50, 750), np.random.randint(50, 550)
        events = []
        base_time = np.random.randint(1000, 100000)
        current_time = base_time
        prev_direction = np.random.uniform(-np.pi, np.pi)
        margin = 30  # Boundary margin

        for i in range(num_events):
            # Calculate time delta based on user type
            if human_type == 'cad_designer':
                time_delta = np.random.normal(180, 30)
                time_delta = max(50, time_delta)
            elif human_type == 'elderly_user':
                time_delta = np.random.gamma(3, 150) + 100
            elif human_type == 'power_user':
                time_delta = np.random.exponential(100) + 30
            else:  # casual_user
                time_delta = np.random.exponential(200) + 50
                
            current_time += int(time_delta)
            
            if i > 0:
                # Movement characteristics
                if human_type == 'cad_designer':
                    direction_change = np.random.normal(0, 0.1)
                    distance = np.random.normal(25, 5)
                elif human_type == 'power_user':
                    direction_change = np.random.normal(0, 0.2)
                    distance = np.random.gamma(2, 12)
                elif human_type == 'elderly_user':
                    direction_change = np.random.normal(0, 0.25)
                    distance = np.random.gamma(1.5, 6)
                else:  # casual_user
                    direction_change = np.random.normal(0, 0.3)
                    distance = np.random.gamma(2, 8)
                
                direction = prev_direction + direction_change
                
                # Calculate new position with boundary handling
                new_x = x + distance * np.cos(direction)
                new_y = y + distance * np.sin(direction)
                
                if new_x < margin or new_x > 800 - margin:
                    direction = np.pi - direction
                    new_x = x + distance * np.cos(direction)
                
                if new_y < margin or new_y > 600 - margin:
                    direction = -direction
                    new_y = y + distance * np.sin(direction)
                
                # Add tremor based on precision
                tremor_magnitude = 3.0 * (1 - precision)
                tremor_x = np.random.normal(0, tremor_magnitude)
                tremor_y = np.random.normal(0, tremor_magnitude)
                
                x = new_x + tremor_x
                y = new_y + tremor_y
                
                # Final boundary check
                x = np.clip(x, 5, 795)
                y = np.clip(y, 5, 595)
                
                prev_direction = direction
                
                # Occasional course corrections
                if np.random.random() < 0.15:
                    prev_direction += np.random.normal(0, 0.8)
            
            # Add pauses based on user type
            if np.random.random() < pause_probability:
                if human_type == 'elderly_user':
                    current_time += np.random.randint(300, 800)
                else:
                    current_time += np.random.randint(150, 500)
            
            events.append({
                'x': round(x, 2),
                'y': round(y, 2),
                'timestamp': current_time
            })
        
        return {
            'session_id': f'human_{session_id}',
            'events': events,
            'time_of_day': np.random.choice(['morning', 'afternoon', 'evening', 'night']),
            'user_type': human_type,
            'label': 'human'
        }
    
    def generate_bot_session(self, session_id: int) -> Dict:
        """Generate bot pointer movement session with simple and advanced (ML trained) bot types."""
        SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
        MARGIN = 50
        
        # Only these two bot types now
        bot_type = np.random.choice(['simple_bot', 'ml_bot'])
        
        # Configure based on bot type
        if bot_type == 'simple_bot':
            num_events = np.random.randint(8, 30)
            pattern = np.random.choice(['linear', 'grid', 'perfect_circle'])
            start_positions = [(0, 0), (100, 100), (200, 200), (50, 50)]
            x, y = random.choice(start_positions)
        else:  # ml_bot
            num_events = np.random.randint(20, 60)
            pattern = 'learned_human'
            x, y = np.random.randint(MARGIN, SCREEN_WIDTH-MARGIN), \
                np.random.randint(MARGIN, SCREEN_HEIGHT-MARGIN)

        events = []
        base_time = np.random.randint(1000, 100000)
        current_time = base_time

        for i in range(num_events):
            prev_x, prev_y = x, y
            
            # Movement patterns
            if pattern == 'linear':
                dx, dy = np.random.choice([10, 20, 30]), np.random.choice([5, 10, 15])
                current_time += 100
            elif pattern == 'grid':
                x = (i % 10) * (SCREEN_WIDTH-MARGIN*2) / 10 + MARGIN
                y = (i // 10) * (SCREEN_HEIGHT-MARGIN*2) / 10 + MARGIN
                current_time += 100
                events.append({'x': x, 'y': y, 'timestamp': current_time})
                continue
            elif pattern == 'perfect_circle':
                angle = (i / num_events) * 2 * np.pi
                radius = min(150, SCREEN_WIDTH//2 - MARGIN, SCREEN_HEIGHT//2 - MARGIN)
                dx = radius * np.cos(angle) - (x - SCREEN_WIDTH//2)
                dy = radius * np.sin(angle) - (y - SCREEN_HEIGHT//2)
                current_time += 50
            else:  # learned_human (ml_bot)
                time_delta = np.random.exponential(180) + 80
                current_time += int(time_delta)
                direction = np.random.normal(0, 0.4)
                distance = np.random.gamma(1.8, 9)
                dx, dy = distance * np.cos(direction), distance * np.sin(direction)
                
                # Add fake tremor (less random than real humans)
                if np.random.random() < 0.3:
                    dx += np.random.normal(0, 0.8)
                    dy += np.random.normal(0, 0.8)

            # Apply movement with boundary handling
            if pattern != 'grid':
                # Redirect if approaching boundaries
                if (x + dx < MARGIN) or (x + dx > SCREEN_WIDTH - MARGIN):
                    dx *= -0.8  # Reverse and dampen
                if (y + dy < MARGIN) or (y + dy > SCREEN_HEIGHT - MARGIN):
                    dy *= -0.8  # Reverse and dampen
                
                x += dx
                y += dy

            # Final position validation
            x = np.clip(x, MARGIN, SCREEN_WIDTH - MARGIN)
            y = np.clip(y, MARGIN, SCREEN_HEIGHT - MARGIN)

            events.append({
                'x': round(x, 2),
                'y': round(y, 2),
                'timestamp': current_time
            })

        return {
            'session_id': f'bot_{session_id}',
            'events': events,
            'time_of_day': np.random.choice(['morning', 'afternoon', 'evening', 'night']),
            'bot_type': bot_type,
            'label': 'bot'
        }
    
    def generate_dataset(self, num_human=50, num_bot=50) -> List[Dict]:
        """Generate complete dataset with human and bot sessions."""
        dataset = []
        
        print(f"Generating {num_human} human sessions...")
        for i in range(num_human):
            dataset.append(self.generate_human_session(i))
        
        print(f"Generating {num_bot} bot sessions...")
        for i in range(num_bot):
            dataset.append(self.generate_bot_session(i))
        
        return dataset
    
    
    def extract_features(self, session: Dict) -> Dict:
        """Extract behavioral features from a session."""
        events = session['events']
        if len(events) < 2:
            return {
                'velocity_mean': 0, 'velocity_std': 0, 'velocity_max': 0,
                'time_interval_mean': 0, 'time_interval_std': 0,
                'direction_entropy': 0, 'path_efficiency': 0,
                'acceleration_variance': 0, 
                'time_morning': 0, 'time_afternoon': 0, 'time_evening': 0, 'time_night': 0
            }
        
        # Calculate velocities
        velocities = []
        time_intervals = []
        directions = []
        
        for i in range(1, len(events)):
            prev_event = events[i-1]
            curr_event = events[i]
            
            # Distance and time
            dx = curr_event['x'] - prev_event['x']
            dy = curr_event['y'] - prev_event['y']
            dt = curr_event['timestamp'] - prev_event['timestamp']
            
            if dt > 0:
                distance = np.sqrt(dx**2 + dy**2)
                velocity = distance / dt * 1000  # pixels per second
                velocities.append(velocity)
                time_intervals.append(dt)
                
                # Direction (angle)
                if distance > 0:
                    direction = np.arctan2(dy, dx)
                    directions.append(direction)
        
        # Feature 1: Velocity Statistics
        velocity_mean = np.mean(velocities) if velocities else 0
        velocity_std = np.std(velocities) if velocities else 0
        velocity_max = np.max(velocities) if velocities else 0
        
        # Feature 2: Time Interval Statistics
        time_interval_mean = np.mean(time_intervals) if time_intervals else 0
        time_interval_std = np.std(time_intervals) if time_intervals else 0
        
        # Feature 3: Direction Entropy (measure of movement predictability)
        direction_entropy = 0
        if directions:
            # Discretize directions into bins
            direction_bins = np.histogram(directions, bins=8, range=(-np.pi, np.pi))[0]
            direction_probs = direction_bins / len(directions)
            direction_probs = direction_probs[direction_probs > 0]  # Remove zero probabilities
            direction_entropy = -np.sum(direction_probs * np.log2(direction_probs))
        
        # Feature 4: Path Efficiency (straight-line distance / actual path length)
        path_efficiency = 0
        if len(events) >= 2:
            start_point = events[0]
            end_point = events[-1]
            straight_distance = np.sqrt((end_point['x'] - start_point['x'])**2 + 
                                      (end_point['y'] - start_point['y'])**2)
            
            total_path_length = sum(np.sqrt((events[i]['x'] - events[i-1]['x'])**2 + 
                                          (events[i]['y'] - events[i-1]['y'])**2) 
                                  for i in range(1, len(events)))
            
            if total_path_length > 0:
                path_efficiency = straight_distance / total_path_length
        
        # Feature 5: Acceleration Variance
        acceleration_variance = 0
        if len(velocities) >= 2:
            accelerations = []
            for i in range(1, len(velocities)):
                if time_intervals[i] > 0:
                    acceleration = (velocities[i] - velocities[i-1]) / time_intervals[i] * 1000
                    accelerations.append(acceleration)
            if accelerations:
                acceleration_variance = np.var(accelerations)
        
        # Metadata feature: Time of day one-hot encoding
        time_of_day = session['time_of_day']
        time_morning = 1 if time_of_day == 'morning' else 0
        time_afternoon = 1 if time_of_day == 'afternoon' else 0
        time_evening = 1 if time_of_day == 'evening' else 0
        time_night = 1 if time_of_day == 'night' else 0
        
        return {
            'velocity_mean': velocity_mean,
            'velocity_std': velocity_std,
            'velocity_max': velocity_max,
            'time_interval_mean': time_interval_mean,
            'time_interval_std': time_interval_std,
            'direction_entropy': direction_entropy,
            'path_efficiency': path_efficiency,
            'acceleration_variance': acceleration_variance,
            'time_morning': time_morning,
            'time_afternoon': time_afternoon,
            'time_evening': time_evening,
            'time_night': time_night
        }
    
    def prepare_training_data(self, dataset: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare feature matrix and labels for training."""
        features_list = []
        labels = []
        
        print("Extracting features from sessions...")
        for session in dataset:
            features = self.extract_features(session)
            features_list.append(list(features.values()))
            labels.append(1 if session['label'] == 'human' else 0)
        
        self.feature_names = list(self.extract_features(dataset[0]).keys())
        return np.array(features_list), np.array(labels)
    
    def train_model(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train and compare multiple models using GridSearchCV."""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Define models and their hyperparameter grids
        models = {
            'RandomForest': {
                'model': RandomForestClassifier(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                }
            },
            'XGBoost': {
                'model': XGBClassifier(random_state=42, eval_metric='logloss'),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0],
                    'reg_alpha': [0, 0.1, 0.5],
                    'reg_lambda': [1, 1.5, 2]
                }
            }
        }
        
        best_model = None
        best_score = 0
        best_model_name = ""
        model_results = {}
        
        print("Training and comparing models with GridSearchCV...")
        
        for model_name, model_config in models.items():
            print(f"\nTraining {model_name}...")
            
            # GridSearchCV with cross-validation
            grid_search = GridSearchCV(
                estimator=model_config['model'],
                param_grid=model_config['params'],
                cv=5,  # 5-fold cross-validation
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
            
            # Fit the grid search
            grid_search.fit(X_train, y_train)
            
            # Get best model
            best_estimator = grid_search.best_estimator_
            
            # Cross-validation scores
            cv_scores = cross_val_score(best_estimator, X_train, y_train, 
                                      cv=5, scoring='roc_auc')
            
            # Test predictions
            y_pred = best_estimator.predict(X_test)
            y_pred_proba = best_estimator.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            avg_precision = average_precision_score(y_test, y_pred_proba)
            
            # Store results
            model_results[model_name] = {
                'best_estimator': best_estimator,
                'best_params': grid_search.best_params_,
                'best_cv_score': grid_search.best_score_,
                'cv_scores': cv_scores,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_roc_auc': roc_auc,
                'test_avg_precision': avg_precision,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'precision': precision,
                'recall': recall
            }
            
            print(f"{model_name} - Best CV Score: {grid_search.best_score_:.4f}")
            print(f"{model_name} - Test ROC-AUC: {roc_auc:.4f}")
            print(f"{model_name} - CV Mean ± Std: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            
            # Track best model
            if roc_auc > best_score:
                best_score = roc_auc
                best_model = best_estimator
                best_model_name = model_name
        
        # Set the best model as the main model
        self.model = best_model
        
        print(f"\nBest Model: {best_model_name} with ROC-AUC: {best_score:.4f}")
        
        # Return results for the best model
        best_results = model_results[best_model_name]
        return {
            'best_model_name': best_model_name,
            'model_results': model_results,
            'roc_auc': best_results['test_roc_auc'],
            'avg_precision': best_results['test_avg_precision'],
            'y_test': y_test,
            'y_pred': best_results['y_pred'],
            'y_pred_proba': best_results['y_pred_proba'],
            'precision': best_results['precision'],
            'recall': best_results['recall'],
            'best_params': best_results['best_params'],
            'cv_scores': best_results['cv_scores']
        }
    
    def generate_report(self, results: Dict) -> None:
        """Generate comprehensive performance report."""
        print("\n" + "="*60)
        print("CAPTCHA BOT DETECTION - PERFORMANCE REPORT")
        print("="*60)
        
        print(f"\nBest Model: {results['best_model_name']}")
        print(f"ROC-AUC Score: {results['roc_auc']:.4f}")
        print(f"Average Precision: {results['avg_precision']:.4f}")
        
        print(f"\nCross-Validation Scores:")
        cv_scores = results['cv_scores']
        print(f"CV Mean: {cv_scores.mean():.4f}")
        print(f"CV Std: {cv_scores.std():.4f}")
        print(f"CV Scores: {[f'{score:.4f}' for score in cv_scores]}")
        
        print(f"\nBest Hyperparameters:")
        for param, value in results['best_params'].items():
            print(f"  {param}: {value}")
        
        print(f"\nModel Comparison:")
        for model_name, model_result in results['model_results'].items():
            print(f"  {model_name}:")
            print(f"    CV Score: {model_result['cv_mean']:.4f} ± {model_result['cv_std']:.4f}")
            print(f"    Test ROC-AUC: {model_result['test_roc_auc']:.4f}")
            print(f"    Test Avg Precision: {model_result['test_avg_precision']:.4f}")
        
        print(f"\nClassification Report:")
        print(classification_report(results['y_test'], results['y_pred'], 
                                  target_names=['Bot', 'Human']))
        
        print(f"\nConfusion Matrix:")
        cm = confusion_matrix(results['y_test'], results['y_pred'])
        print(cm)
        
        # Feature importance
        if self.model:
            print(f"\nFeature Importance:")
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                for _, row in feature_importance.iterrows():
                    print(f"  {row['feature']}: {row['importance']:.4f}")
            else:
                print("  Feature importance not available for this model.")
    
    def visualize_results(self, results: Dict) -> None:
        """Create visualizations for model performance."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # ROC Curve
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(results['y_test'], results['y_pred_proba'])
        axes[0, 0].plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {results["roc_auc"]:.4f})')
        axes[0, 0].plot([0, 1], [0, 1], 'k--')
        axes[0, 0].set_xlabel('False Positive Rate')
        axes[0, 0].set_ylabel('True Positive Rate')
        axes[0, 0].set_title(f'ROC Curve - {results["best_model_name"]}')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Precision-Recall Curve
        axes[0, 1].plot(results['recall'], results['precision'], linewidth=2, 
                       label=f'PR Curve (AP = {results["avg_precision"]:.4f})')
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Precision-Recall Curve')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Cross-Validation Scores
        cv_scores = results['cv_scores']
        axes[0, 2].bar(range(len(cv_scores)), cv_scores)
        axes[0, 2].axhline(y=cv_scores.mean(), color='r', linestyle='--', 
                          label=f'Mean: {cv_scores.mean():.4f}')
        axes[0, 2].set_xlabel('CV Fold')
        axes[0, 2].set_ylabel('ROC-AUC Score')
        axes[0, 2].set_title('Cross-Validation Scores')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # Feature Importance
        if self.model and hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            axes[1, 0].barh(feature_importance['feature'], feature_importance['importance'])
            axes[1, 0].set_xlabel('Importance')
            axes[1, 0].set_title('Feature Importance')
        else:
            axes[1, 0].text(0.5, 0.5, 'Feature importance\nnot available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Feature Importance')
            
        # Confusion Matrix Heatmap
        cm = confusion_matrix(results['y_test'], results['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Bot', 'Human'], yticklabels=['Bot', 'Human'],
                   ax=axes[1, 1])
        axes[1, 1].set_title('Confusion Matrix')
        axes[1, 1].set_xlabel('Predicted')
        axes[1, 1].set_ylabel('Actual')
        
        # Model Comparison
        model_names = list(results['model_results'].keys())
        test_scores = [results['model_results'][name]['test_roc_auc'] for name in model_names]
        cv_scores_mean = [results['model_results'][name]['cv_mean'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        axes[1, 2].bar(x - width/2, test_scores, width, label='Test ROC-AUC', alpha=0.8)
        axes[1, 2].bar(x + width/2, cv_scores_mean, width, label='CV Mean ROC-AUC', alpha=0.8)
        axes[1, 2].set_xlabel('Models')
        axes[1, 2].set_ylabel('ROC-AUC Score')
        axes[1, 2].set_title('Model Comparison')
        axes[1, 2].set_xticks(x)
        axes[1, 2].set_xticklabels(model_names)
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def main():
    """Main execution function."""
    # Initialize detector
    detector = CAPTCHABotDetector()
    
    # Generate dataset
    print("Starting CAPTCHA Bot Detection System...")
    dataset = detector.generate_dataset(num_human=50, num_bot=50)
    
    # Prepare training data
    X, y = detector.prepare_training_data(dataset)
    
    # Train model and evaluate
    results = detector.train_model(X, y)
    
    # Generate report
    detector.generate_report(results)
    
    # Visualize results
    detector.visualize_results(results)
    
    print("\nSystem ready for production deployment!")
    
    # Example of how to use the trained model for prediction
    print("\nExample prediction on new session:")
    new_session = detector.generate_human_session(999)
    new_features = detector.extract_features(new_session)
    new_X = np.array(list(new_features.values())).reshape(1, -1)
    
    prediction = detector.model.predict(new_X)[0]
    probability = detector.model.predict_proba(new_X)[0]
    
    print(f"Session ID: {new_session['session_id']}")
    print(f"Prediction: {'Human' if prediction == 1 else 'Bot'}")
    print(f"Confidence: {max(probability):.4f}")

if __name__ == "__main__":
    main()