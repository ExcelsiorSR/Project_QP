# ======================================================
#                  MODEL IMPORTS
# ======================================================

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score, f1_score, recall_score
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# ========================================================
#                 MODEL TRAINING MODULE
# ========================================================

class CostSensitiveEnsemble:
    """
    Ensemble with cost-sensitive learning
    Optimized to minimize False Negatives (missing crashes)
    """
    
    def __init__(self, fn_cost=5, fp_cost=3):
        """
        fn_cost: Cost of missing a crash (False Negative) - HIGH
        fp_cost: Cost of false alarm (False Positive) - LOW
        
        For crash prediction, missing a crash is 10x worse than a false alarm
        """
        self.models = {}
        self.scaler = StandardScaler()
        self.fn_cost = fn_cost
        self.fp_cost = fp_cost
        self.optimal_threshold = 0.5

    # ==========================================
    # TRAINS ENSEMBLE
    # ==========================================    
    def train(self, X_train, y_train):

        
        print("\n" + "="*60)
        print("🎯 TRAINING COST-SENSITIVE ENSEMBLE")
        print("="*60)
        print(f"False Negative Cost: {self.fn_cost}x")
        print(f"False Positive Cost: {self.fp_cost}x\n")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Calculate extreme class weight for crash detection
        n_normal = np.sum(y_train == 0)
        n_crash = np.sum(y_train == 1)
        
        # Apply cost-sensitive weighting
        crash_weight = (n_normal / n_crash) * (self.fn_cost / self.fp_cost)
        
        print(f"📊 Class distribution:")
        print(f"   Normal days: {n_normal}")
        print(f"   Crash days: {n_crash}")
        print(f"   Adjusted crash weight: {crash_weight:.2f}\n")
        
        # =========================================
        # MODEL 1: XGBoost with extreme weighting
        # =========================================
        
        print("📊 Training XGBoost...")
        self.models['xgb'] = xgb.XGBClassifier(
            base_score = 0.5,
            scale_pos_weight=crash_weight,
            max_depth=3,  # Shallow to avoid overfitting
            learning_rate=0.005,  # Very low learning rate
            n_estimators=500,  # More trees to compensate
            min_child_weight=1,  # Allow smaller leaf nodes
            subsample=0.7,
            colsample_bytree=0.7,
            gamma=0.1,  # Lower gamma = more aggressive splits
            reg_alpha=0.05,
            reg_lambda=0.5,
            random_state=42,
            eval_metric='aucpr'  # Focus on precision-recall
        )
        self.models['xgb'].fit(X_train_scaled, y_train)
        
        # ==============================================
        # MODEL 2: Random Forest with balanced weight
        # ==============================================
        
        print("🌲 Training Random Forest...")
        self.models['rf'] = RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_split=5,  # Lower to capture rare crash patterns
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight={0: 1, 1: crash_weight},  # Custom weights
            random_state=42,
            n_jobs=-1
        )
        self.models['rf'].fit(X_train_scaled, y_train)
        
        # ========================================
        # MODEL 3: Gradient Boosting
        # ========================================
        
        print("📈 Training Gradient Boosting...")
        
        # Create sample weights for cost-sensitive learning
        sample_weights = np.where(y_train == 1, crash_weight, 1.0)
        
        self.models['gb'] = GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.005,
            max_depth=3,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.7,
            random_state=42
        )
        self.models['gb'].fit(X_train_scaled, y_train, sample_weight=sample_weights)
        
        # ========================================
        # MODEL 4: Logistic Regression
        # ========================================
        
        print("📉 Training Logistic Regression...")
        self.models['lr'] = LogisticRegression(
            class_weight={0: 1, 1: crash_weight},
            max_iter=2000,
            random_state=42,
            solver='saga'
        )
        self.models['lr'].fit(X_train_scaled, y_train)
        
        print("\n✅ All models trained with cost-sensitive learning!")
        
        return self
    
    # ================================================
    # GETS ENSEMBLE PREDICTIONS WITH WEIGHTED VOTING
    # ================================================
    def predict_proba(self, X_test):
        """Get ensemble predictions with weighted voting"""
        
        X_test_scaled = self.scaler.transform(X_test)
        
        # Get probabilities from each model
        probs = {}
        for name, model in self.models.items():
            probs[name] = model.predict_proba(X_test_scaled)[:, 1]
        
        # Weighted ensemble - favor models that catch crashes better
        ensemble_probs = (
            0.45 * probs['xgb'] +      # XGBoost best for complex patterns
            0.30 * probs['rf'] +       # RF good for non-linear
            0.20 * probs['gb'] +       # GB good for sequential
            0.05 * probs['lr']         # LR as baseline
        )
        
        return ensemble_probs, probs
    
    # ==========================================
    # FINDS THRESHOLD FOR A CRASH
    # ==========================================
    def find_cost_sensitive_threshold(self, X_val, y_val):
        """
        Finds a threshold that minimizes cost-weighted errors
        Prioritizes catching crashes (minimizes False Negatives)
        """
        
        ensemble_probs, _ = self.predict_proba(X_val)
        
        # GenerateS precision-recall curve
        precisions, recalls, thresholds = precision_recall_curve(y_val, ensemble_probs)
        
        # RemoveS last element (corresponds to undefined threshold)
        precisions = precisions[:-1]
        recalls = recalls[:-1]
        
        print("\n" + "="*60)
        print("🎯 THRESHOLD OPTIMIZATION")
        print("="*60 + "\n")
        
        # Method 1: Cost-based optimization
        best_cost = float('inf')
        best_threshold_cost = 0.5
        
        for threshold, precision, recall in zip(thresholds, precisions, recalls):
            # Estimate false negatives and false positives
            preds = (ensemble_probs >= threshold).astype(int)
            cm = confusion_matrix(y_val, preds)
            
            if cm.shape == (2, 2):
                fn = cm[1, 0]  # False Negatives (missing crashes)
                fp = cm[0, 1]  # False Positives (false alarms)
                
                # Calculate weighted cost
                total_cost = (fn * self.fn_cost) + (fp * self.fp_cost)
                
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_threshold_cost = threshold
        
        # Method 2: High recall threshold (catch 80%+ of crashes)
        target_recall = 0.80
        idx_high_recall = np.where(recalls >= target_recall)[0]
        
        if len(idx_high_recall) > 0:
            # Among high-recall thresholds, pick highest precision
            best_idx = idx_high_recall[np.argmax(precisions[idx_high_recall])]
            best_threshold_recall = thresholds[best_idx]
            
            print(f"High-Recall Threshold (80%+ recall): {best_threshold_recall:.3f}")
            print(f"  Projected Precision: {precisions[best_idx]:.3f}")
            print(f"  Projected Recall: {recalls[best_idx]:.3f}\n")
        else:
            best_threshold_recall = thresholds[np.argmax(recalls)]
            print("⚠️  Could not achieve 80% recall, using max recall threshold\n")
        
        # Method 3: Mathematical equilibrium
        differences = np.abs(precisions - recalls)
        equilibrium_idx = np.argmin(differences)
        best_threshold_eq = thresholds[equilibrium_idx]
        
        print(f"Cost-Optimized Threshold: {best_threshold_cost:.3f}")
        print(f"  Estimated validation cost: {best_cost:.2f}\n")
        
        print(f"Equilibrium Threshold: {best_threshold_eq:.3f}")
        print(f"  Precision = Recall ≈ {precisions[equilibrium_idx]:.3f}\n")
        
        # Choose the cost-optimized threshold for production
        self.optimal_threshold = best_threshold_cost
        
        print(f"🎯 SELECTED THRESHOLD: {self.optimal_threshold:.3f}")
        print("   (Cost-optimized to minimize missed crashes)\n")
        
        return self.optimal_threshold
    
    # ==========================================
    # GENERATES EVALUATION METRICS
    # ==========================================
    def evaluate(self, X_test, y_test, threshold=None):

        
        if threshold is None:
            threshold = self.optimal_threshold
        
        ensemble_probs, individual_probs = self.predict_proba(X_test)
        predictions = (ensemble_probs >= threshold).astype(int)
        
        print("\n" + "="*60)
        print("📊 FINAL TEST SET EVALUATION")
        print("="*60 + "\n")
        
        print(f"Using threshold: {threshold:.3f}\n")
        
        # Classification report
        print("Classification Report:")
        print(classification_report(
            y_test, predictions,
            target_names=['Normal', 'Crash'],
            digits=3
        ))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, predictions)
        print("\nConfusion Matrix:")
        print(cm)
        print(f"\nTrue Negatives: {cm[0,0]} | False Positives: {cm[0,1]}")
        print(f"False Negatives: {cm[1,0]} | True Positives: {cm[1,1]}")
        
        # Calculate cost
        if cm.shape == (2, 2):
            fn = cm[1, 0]
            fp = cm[0, 1]
            total_cost = (fn * self.fn_cost) + (fp * self.fp_cost)
            print(f"\nWeighted Cost: {total_cost:.2f}")
            print(f"  FN Cost: {fn} × {self.fn_cost} = {fn * self.fn_cost}")
            print(f"  FP Cost: {fp} × {self.fp_cost} = {fp * self.fp_cost}")
        
        # Metrics
        roc_auc = roc_auc_score(y_test, ensemble_probs)
        avg_precision = average_precision_score(y_test, ensemble_probs)
        
        print(f"\nROC-AUC Score: {roc_auc:.3f}")
        print(f"Average Precision Score: {avg_precision:.3f}")
        
        # Crash detection performance
        if np.sum(y_test == 1) > 0:
            crash_recall = recall_score(y_test, predictions)
            print(f"\n🎯 Crash Detection Recall: {crash_recall:.1%}")
            print(f"   (Caught {int(crash_recall * np.sum(y_test == 1))}/{int(np.sum(y_test == 1))} crashes)")
        
        return predictions, ensemble_probs
    
    # ==========================================
    # PLOTS PRECISION RECALL TRADEOFF
    # ==========================================
    def plot_threshold_analysis(self, X_val, y_val, save_path='threshold_analysis.png'):
        
        
        ensemble_probs, _ = self.predict_proba(X_val)
        precisions, recalls, thresholds = precision_recall_curve(y_val, ensemble_probs)
        
        # Calculate F1 scores
        f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
        
        plt.figure(figsize=(12, 5))
        
        # Plot 1: Precision-Recall Curve
        plt.subplot(1, 2, 1)
        plt.plot(recalls, precisions, linewidth=2, label='PR Curve')
        plt.axhline(y=average_precision_score(y_val, ensemble_probs), 
                    color='r', linestyle='--', label=f'Avg Precision')
        plt.xlabel('Recall (Crash Detection Rate)')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Tradeoff')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Metrics vs Threshold
        plt.subplot(1, 2, 2)
        plt.plot(thresholds, precisions[:-1], label='Precision', linewidth=2)
        plt.plot(thresholds, recalls[:-1], label='Recall', linewidth=2)
        plt.plot(thresholds, f1_scores, label='F1 Score', linewidth=2, linestyle='--')
        plt.axvline(x=self.optimal_threshold, color='r', linestyle=':', 
                    label=f'Optimal ({self.optimal_threshold:.3f})')
        plt.xlabel('Classification Threshold')
        plt.ylabel('Score')
        plt.title('Metrics vs Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Threshold analysis saved to {save_path}")
    
    # ==========================================
    # PLOTS AGGREGATED FEATURE IMPORTANCE
    # ==========================================
    def plot_feature_importance(self, feature_names, top_n=20, save_path='feature_importance.png'):
        
        
        # Get importance from tree models
        xgb_imp = self.models['xgb'].feature_importances_
        rf_imp = self.models['rf'].feature_importances_
        
        # Weighted average
        avg_importance = (0.5 * xgb_imp + 0.5 * rf_imp)
        
        # Create dataframe
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': avg_importance
        }).sort_values('Importance', ascending=False).head(top_n)
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df, y='Feature', x='Importance', palette='rocket')
        plt.title(f'Top {top_n} Most Important Features for Crash Prediction')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Feature importance saved to {save_path}")
        
        return importance_df